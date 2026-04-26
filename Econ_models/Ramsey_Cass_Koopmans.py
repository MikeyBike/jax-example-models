from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot  as plt

import equinox as eqx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)



# Model container

class RCK_model(eqx.Module):
    '''Ramsey-Cass-Koopmans perfect-foresight transition model'''

    alpha: float
    delta: float
    beta: float
    n: float
    g: float
    T: int = eqx.field(static=True)

    A_path: jnp.ndarray
    L_path: jnp.ndarray
    K0: float
    CTp1: float
    YTp1: float


    @property
    def growth_agg(self) -> float:
        return self.n + self.g + self.n * self.g
    
    def k_star(self) -> float:
        """Captital per effective worker at the BGP"""
        num = (1.0 / self.beta) + (1.0 + self.n) * (1.0 + self.g) - (1.0 - self.delta)
        return float((num / self.alpha) ** (1.0 / (self.alpha - 1.0)))
    
    def bgp_values(self, A: float, L: float) -> Tuple[float, float, float, float]:
        """Aggregate BGP value at a given (A, L)."""
        k = self.k_star()
        K = A * L * (1.0 + self.n) * (1.0 + self.g) * k
        K_lag = K / ((1.0 + self.n) * (1.0 + self.g))
        Y = K_lag ** self.alpha * (A * L) ** (1.0 - self.alpha)
        Inv = (1.0 - (1.0 - self.delta) / ((1.0 + self.n) * (1.0 + self.g))) * K 
        C = Y - Inv
        return float(K), float(Y), float(C), float(Inv)
    


    # Residuals 

    def _local_residual(
            self,
            x_prev: jnp.ndarray,
            x_curr: jnp.ndarray,
            x_next: jnp.ndarray,
            t: int,
    ) -> jnp.ndarray:
        """ Residual block at time t 

        Each x_t has shape (4, ) with entries [C_t, K_t, Y_t, I_t]
        """

        C_t, K_t, Y_t, I_t = x_curr

        # Boundary values:
        # K_{t-1} is fixed to K_0 at t=0
        # C_{t+1} and Y_{t+1} are fixed to terminal BGP values at t=T-1

        K_lag = jnp.where(t == 0, jnp.asarray(self.K0), x_prev[1])
        C_lead = jnp.where(t == self.T -1, jnp.asarray(self.CTp1), x_next[0])
        Y_lead = jnp.where(t == self.T -1, jnp.asarray(self.YTp1), x_next[2])

        A_t = self.A_path[t]
        L_t = self.L_path[t]

        r1 = K_t - (1.0 - self.delta) * K_lag - I_t
        r2 = I_t + C_t - Y_t
        r3 = 1.0 / C_t - self.beta / C_lead * (
            self.alpha * Y_lead / K_t + 1.0 - self.delta
        )
        r4 = Y_t - K_lag ** self.alpha * (A_t * L_t) ** (1.0 - self.alpha)


        return jnp.asarray([r1, r2, r3, r4], dtype=jnp.float64)
    
    
    def residual_blocks(self, X: jnp.ndarray) -> jnp.ndarray:
        """Return residuals with shape (T, 4)"""

        X_prev = jnp.concatenate([X[:1], X[:-1]], axis=0)
        X_next = jnp.concatenate([X[1:], X[-1:]], axis=0)
        ts = jnp.arange(self.T)

        def one_residual(args):
            xp, xc, xn, t = args
            return self._local_residual(xp, xc, xn, t)
        
        return jax.vmap(one_residual)((X_prev, X, X_next, ts))
    
    def residual_vector(self, x_flat: jnp.ndarray) -> jnp.ndarray:
        """Flattened residual vector of length 4T"""
        
        X = x_flat.reshape((self.T, 4))
        return self.residual_blocks(X).reshape((-1, ))
    
    def block_jacobian(self, X: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return the block tri-diagonal Jacobian pieces
        
        A[t] = dr_t / dx_{t-1}
        B[t] = dr_t / dx_t
        C[t] = dr_t / dx_{t+1}

        """

        X_prev = jnp.concatenate([X[:1], X[:-1]], axis=0)
        X_next = jnp.concatenate([X[1:], X[-1:]], axis=0)
        ts = jnp.arange(self.T)

        jac_fn = jax.jacfwd(self._local_residual, argnums=(0, 1, 2))

        def one_jac(args):
            xp, xc, xn, t = args
            j_prev, j_curr, j_next = jac_fn(xp, xc, xn, t)
            j_prev, j_curr, j_next = jac_fn(xp, xc, xn, t)
            return j_prev, j_curr, j_next
        
        A, B, C = jax.vmap(one_jac)((X_prev, X, X_next, ts))
        return A, B, C

    
    @staticmethod
    def solve_block_tridiagonal(A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, d: jnp.ndarray,) -> jnp.ndarray:
        """Solve a block tridiagonal linear system.

        System:
            A_t x_{t-1} + B_t x_t + C_t x_{t+1} = d_t

        with A_0 unused and C_{T-1} unused.

        Args:
        A, B, C : arrays of shape (T, m, m)
        d       : array of shape (T, m)

        Returns:
        x : array of shape (T, m)
        """
        T = B.shape[0]
        m = B.shape[1]

        # Forward elimination
        Bm = []
        dm = []

        Bm0 = B[0]
        dm0 = d[0]
        Bm.append(Bm0)
        dm.append(dm0)

        for t in range(1, T):
            L_t = jax.scipy.linalg.solve(Bm[t - 1].T, A[t].T, assume_a="gen").T
            B_t = B[t] - L_t @ C[t - 1]
            d_t = d[t] - L_t @ dm[t - 1]
            Bm.append(B_t)
            dm.append(d_t)

        # Back substitution.
        x = [None] * T
        x[T - 1] = jax.scipy.linalg.solve(Bm[T - 1], dm[T - 1], assume_a="gen")
        for t in range(T - 2, -1, -1):
            rhs = dm[t] - C[t] @ x[t + 1]
            x[t] = jax.scipy.linalg.solve(Bm[t], rhs, assume_a="gen")

        return jnp.stack(x, axis=0)

    

