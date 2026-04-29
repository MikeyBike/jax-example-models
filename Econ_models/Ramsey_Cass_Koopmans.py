"""Ramsey-Cass-Koopmans Perfect Foresight Model
================================================
A JAX / Equinox implementation with a block-tridiagonal Newton solver.

Basic economic setup is:
  - Cobb-Douglas production
  - Capital accumulation
  - Resource constraint
  - Euler equation
  - Perfect foresight transition path to the BGP
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)



# -----------------------------------------------------------------------------
# Model container
# -----------------------------------------------------------------------------


class RCK_model(eqx.Module):
    """Ramsey-Cass-Koopmans perfect-foresight transition model"""

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
        """Capital per effective worker at the BGP"""
        num = (1.0 / self.beta) * (1.0 + self.n) * (1.0 + self.g) - (1.0 - self.delta)
        return float((num / self.alpha) ** (1.0 / (self.alpha - 1.0)))
    
    # Dynastic version, uncomment r3 too
    # def k_star(self) -> float:
    #     growth = (1.0 + self.n) * (1.0 + self.g)
    #     num = self.alpha * growth ** (1.0 - self.alpha)
    #     den = (1.0 + self.g) / self.beta - (1.0 - self.delta)
    #     return float((num / den) ** (1.0 / (1.0 - self.alpha)))

    def bgp_values(self, A: float, L: float) -> Tuple[float, float, float, float]:
        """Aggregate BGP values at a given (A, L)"""
        k = self.k_star()
        K = A * L * (1.0 + self.n) * (1.0 + self.g) * k
        Klag = K / ((1.0 + self.n) * (1.0 + self.g))
        Y = Klag ** self.alpha * (A * L) ** (1.0 - self.alpha)
        Inv = (1.0 - (1.0 - self.delta) / ((1.0 + self.n) * (1.0 + self.g))) * K
        C = Y - Inv
        return float(K), float(Y), float(C), float(Inv)

    # ------------------------------------------------------------------
    # Residual blocks
    # ------------------------------------------------------------------

    def local_residual(
        self,
        x_prev: jnp.ndarray,
        x_curr: jnp.ndarray,
        x_next: jnp.ndarray,
        t: int,
    ) -> jnp.ndarray:
        """Residual block at time t

        Each x_t has shape (4,) with entries [C_t, K_t, Y_t, I_t]
        """
        C_t, K_t, Y_t, I_t = x_curr

        # Boundary handling:
        #   K_{t-1} is fixed to K0 at t=0
        #   C_{t+1} and Y_{t+1} are fixed to terminal BGP values at t=T-1
        K_lag = jnp.where(t == 0, jnp.asarray(self.K0), x_prev[1])
        C_lead = jnp.where(t == self.T - 1, jnp.asarray(self.CTp1), x_next[0])
        Y_lead = jnp.where(t == self.T - 1, jnp.asarray(self.YTp1), x_next[2])

        A_t = self.A_path[t]
        L_t = self.L_path[t]

        r1 = K_t - (1.0 - self.delta) * K_lag - I_t
        r2 = I_t + C_t - Y_t
        
        r3 = 1.0 / C_t - self.beta / C_lead * (self.alpha * Y_lead / K_t + 1.0 - self.delta) 
        """Uncomment k_star and r3 below to see the dynasty-weighted aggregate utility, which is more more patient when population grows (i.e. more saving, capital accumulation, and higher k_star)."""
        # r3 = 1.0 / C_t - self.beta * (1.0 + self.n) / C_lead * (self.alpha * Y_lead / K_t + 1.0 - self.delta)  

        r4 = Y_t - K_lag ** self.alpha * (A_t * L_t) ** (1.0 - self.alpha)

        return jnp.array([r1, r2, r3, r4], dtype=jnp.float64)

    def residual_blocks(self, X: jnp.ndarray) -> jnp.ndarray:
        """Return residuals with shape (T, 4)."""
        X_prev = jnp.concatenate([X[:1], X[:-1]], axis=0)
        X_next = jnp.concatenate([X[1:], X[-1:]], axis=0)
        ts = jnp.arange(self.T)

        def one_residual(args):
            xp, xc, xn, t = args
            return self.local_residual(xp, xc, xn, t)

        return jax.vmap(one_residual)(
            (X_prev, X, X_next, ts)
        )

    def residual_vector(self, x_flat: jnp.ndarray) -> jnp.ndarray:
        """Flattened residual vector of length 4T."""
        X = x_flat.reshape((self.T, 4))
        return self.residual_blocks(X).reshape((-1,))

    def block_jacobians(
        self, X: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return the block tridiagonal Jacobian pieces

        A[t] = d r_t / d x_{t-1}
        B[t] = d r_t / d x_t
        C[t] = d r_t / d x_{t+1}

        Shapes:
          A, B, C: (T, 4, 4)
        """
        X_prev = jnp.concatenate([X[:1], X[:-1]], axis=0)
        X_next = jnp.concatenate([X[1:], X[-1:]], axis=0)
        ts = jnp.arange(self.T)

        jac_fn = jax.jacfwd(self.local_residual, argnums=(0, 1, 2))

        def one_jac(args):
            xp, xc, xn, t = args
            j_prev, j_curr, j_next = jac_fn(xp, xc, xn, t)
            return j_prev, j_curr, j_next

        A, B, C = jax.vmap(one_jac)((X_prev, X, X_next, ts))
        return A, B, C

    # ------------------------------------------------------------------
    # Block tridiagonal solve
    # ------------------------------------------------------------------

    @staticmethod
    def solve_block_tridiagonal(
        A: jnp.ndarray,
        B: jnp.ndarray,
        C: jnp.ndarray,
        d: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve a block tridiagonal linear system

        System:
            A_t x_{t-1} + B_t x_t + C_t x_{t+1} = d_t

        with A_0 unused and C_{T-1} unused

        Args
        ----
        A, B, C : arrays of shape (T, m, m)
        d       : array of shape (T, m)

        Returns
        -------
        x : array of shape (T, m)
        """
        T = B.shape[0]
        m = B.shape[1]

        # Forward elimination.
        Bm = []
        dm = []

        Bm0 = B[0]
        dm0 = d[0]
        Bm.append(Bm0)
        dm.append(dm0)

        for t in range(1, T):
            # L_t = A_t @ inv(Bm[t-1])
            L_t = jax.scipy.linalg.solve(Bm[t - 1].T, A[t].T, assume_a="gen").T
            B_t = B[t] - L_t @ C[t - 1]
            d_t = d[t] - L_t @ dm[t - 1]
            Bm.append(B_t)
            dm.append(d_t)

        # Back substitution
        x = [None] * T
        x[T - 1] = jax.scipy.linalg.solve(Bm[T - 1], dm[T - 1], assume_a="gen")
        for t in range(T - 2, -1, -1):
            rhs = dm[t] - C[t] @ x[t + 1]
            x[t] = jax.scipy.linalg.solve(Bm[t], rhs, assume_a="gen")

        return jnp.stack(x, axis=0)

    # ------------------------------------------------------------------
    # Newton step and solver
    # ------------------------------------------------------------------

    def newton_direction(self, X: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute Newton direction dx and current residual blocks"""
        F = self.residual_blocks(X)
        A, B, C = self.block_jacobians(X)
        rhs = -F
        dx = self.solve_block_tridiagonal(A, B, C, rhs)
        return dx, F

    def _residual_norm(self, X: jnp.ndarray) -> jnp.ndarray:
        F = self.residual_blocks(X)
        return jnp.max(jnp.abs(F))

    def line_search(
        self,
        X: jnp.ndarray,
        dx: jnp.ndarray,
        current_norm: float,
        max_backtracks: int = 30,
        shrink: float = 0.5,
    ) -> Tuple[jnp.ndarray, float]:
        
        step = 1.0
        X_best = X
        best_norm = current_norm

        for _ in range(max_backtracks):
            X_cand = X + step * dx
            cand_norm = float(self._residual_norm(X_cand))
            if cand_norm < best_norm:
                X_best = X_cand
                best_norm = cand_norm
                break
            step *= shrink

        return X_best, best_norm

    def solve(
        self,
        X0: jnp.ndarray,
        tol: float = 1e-10,
        max_iter: int = 60,
        verbose: bool = True,
    ) -> jnp.ndarray:
        """Solve for the transition path using damped Newton iterations"""
        _ = self.residual_blocks(X0).block_until_ready()
        _ = self.block_jacobians(X0)

        X = X0
        prev_step = 1.0

        if verbose:
            header = f"{'Iter':>5}  {'max|F|':>12}  {'step':>8}"
            print(header)
            print("-" * len(header))

        for it in range(max_iter):
            F = self.residual_blocks(X)
            norm = float(jnp.max(jnp.abs(F)))
            if verbose:
                print(f"{it:5d}  {norm:12.4e}  {prev_step:8.4f}")
            if norm < tol:
                if verbose:
                    print(f"\n✓ Converged in {it} iterations")
                return X

            dx, _ = self.newton_direction(X)
            X, new_norm = self.line_search(X, dx, norm)
            prev_step = float(jnp.nan_to_num(new_norm / norm, nan=0.0, posinf=0.0))

        if verbose:
            print(f"\n Did not converge after {max_iter} iterations")
        return X


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def growth_rate(X: np.ndarray, scale: Optional[np.ndarray] = None) -> np.ndarray:
    """Period-over-period growth rate, optionally of X / scale"""
    Xd = X if scale is None else X / scale
    return np.diff(Xd) / Xd[:-1]


def build_initial_guess(model: RCK_model) -> jnp.ndarray:
    """Construct a good initial guess that satisfies several equations exactly"""
    T = model.T

    A_np = np.asarray(model.A_path)
    L_np = np.asarray(model.L_path)

    K_steady, Y0_bgp, C0_bgp, I0_bgp = model.bgp_values(1.0, 1.0)
    K_Tp1, _, _, _ = model.bgp_values((1.0 + model.g) ** (T + 1), (1.0 + model.n) ** (T + 1))

    K_init = np.linspace(model.K0, K_Tp1, T)
    K_lag = np.concatenate([[model.K0], K_init[:-1]])
    Y_init = K_lag ** model.alpha * (A_np * L_np) ** (1.0 - model.alpha)
    bgp_IY = I0_bgp / Y0_bgp
    I_init = bgp_IY * Y_init
    C_init = Y_init - I_init

    X0 = np.stack([C_init, K_init, Y_init, I_init], axis=1)
    return jnp.asarray(X0, dtype=jnp.float64)


def unpack_solution(model: RCK_model, X: jnp.ndarray):
    """Return solution arrays with t=0 and terminal t=T+1 values included"""
    T = model.T

    X_np = np.asarray(X)
    C_sol = X_np[:, 0]
    K_sol = X_np[:, 1]
    Y_sol = X_np[:, 2]
    I_sol = X_np[:, 3]

    K_m1 = model.K0 / ((1.0 + model.n) * (1.0 + model.g))
    Y0 = K_m1 ** model.alpha * (1.0 * 1.0) ** (1.0 - model.alpha)
    I0 = (1.0 - (1.0 - model.delta) / ((1.0 + model.n) * (1.0 + model.g))) * model.K0
    C0 = Y0 - I0

    K_Tp1, Y_Tp1, C_Tp1, I_Tp1 = model.bgp_values(
        (1.0 + model.g) ** (T + 1), (1.0 + model.n) ** (T + 1)
    )

    t_full = np.arange(0, T + 2)
    C_full = np.concatenate([[C0], C_sol, [C_Tp1]])
    K_full = np.concatenate([[model.K0], K_sol, [K_Tp1]])
    Y_full = np.concatenate([[Y0], Y_sol, [Y_Tp1]])
    I_full = np.concatenate([[I0], I_sol, [I_Tp1]])
    A_full = np.concatenate([[1.0], np.asarray(model.A_path), [(1.0 + model.g) ** (T + 1)]])
    L_full = np.concatenate([[1.0], np.asarray(model.L_path), [(1.0 + model.n) ** (T + 1)]])

    return t_full, C_full, K_full, Y_full, I_full, A_full, L_full




def make_plot(
    model: RCK_model,
    t_full: np.ndarray,
    C_full: np.ndarray,
    K_full: np.ndarray,
    Y_full: np.ndarray,
    I_full: np.ndarray,
    A_full: np.ndarray,
    L_full: np.ndarray,
    out_path: Optional[str] = None,
):
    BLUE, ORG, GRN, PURP = "#2563EB", "#EA580C", "#16A34A", "#7C3AED"
    GRAY = "#6B7280"

    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor("#F8FAFC")
    fig.suptitle(
        "Ramsey-Cass-Koopmans — Transition to BGP  [JAX · Equinox · Block Newton]",
        fontsize=13,
        fontweight="bold",
        y=0.99,
        color="#1E293B",
    )
    gs = gridspec.GridSpec(
        3, 2, hspace=0.5, wspace=0.35, left=0.07, right=0.97, top=0.94, bottom=0.06
    )

    def stylise(ax, title, xlabel="Period"):
        ax.set_title(title, fontsize=10, fontweight="semibold", color="#1E293B", pad=6)
        ax.set_xlabel(xlabel, fontsize=8, color=GRAY)
        ax.tick_params(labelsize=8, colors=GRAY)
        ax.set_facecolor("#F1F5F9")
        ax.grid(color="white", linewidth=1.2, alpha=0.9)
        for spine in ax.spines.values():
            spine.set_visible(False)

    t_gr = np.arange(1, model.T + 2)
    gK_agg = growth_rate(K_full)
    gK_pc = growth_rate(K_full, L_full)
    gK_int = growth_rate(K_full, A_full * L_full)
    gY_agg = growth_rate(Y_full)
    gY_pc = growth_rate(Y_full, L_full)
    gY_int = growth_rate(Y_full, A_full * L_full)

    # (1) Log levels
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_full, np.log(K_full), lw=2.2, color=PURP, label="log K")
    ax1.plot(t_full, np.log(Y_full), lw=2.2, color=BLUE, label="log Y")
    ax1.plot(t_full, np.log(C_full), lw=2.2, color=GRN, label="log C")
    bgp_slope = np.log(1.0 + model.growth_agg)
    t_ref = np.linspace(0, model.T + 1, 200)
    ref_K = np.log(K_full[0]) + bgp_slope * t_ref
    ax1.plot(t_ref, ref_K, lw=1.2, color=GRAY, ls="--", alpha=0.6, label="BGP slope")
    stylise(ax1, "Log Levels — converging to the BGP slope")
    ax1.legend(fontsize=8, framealpha=0.9, ncol=4)

    # (2) Capital growth rates
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_gr, gK_agg, lw=2, color=BLUE, label="Aggregate")
    ax2.plot(t_gr, gK_pc, lw=2, color=GRN, label="Per capita")
    ax2.plot(t_gr, gK_int, lw=2, color=ORG, label="Intensive form")
    ax2.axhline(model.growth_agg, ls="--", lw=1.2, color=BLUE, alpha=0.5, label="BGP agg")
    ax2.axhline(model.g, ls="--", lw=1.2, color=GRN, alpha=0.5, label="BGP pc")
    ax2.axhline(0.0, ls="--", lw=1.2, color=ORG, alpha=0.5, label="BGP int")
    stylise(ax2, "Capital Growth Rates")
    ax2.legend(fontsize=7, framealpha=0.9)

    # (3) Output growth rates
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t_gr, gY_agg, lw=2, color=BLUE, label="Aggregate")
    ax3.plot(t_gr, gY_pc, lw=2, color=GRN, label="Per capita")
    ax3.plot(t_gr, gY_int, lw=2, color=ORG, label="Intensive form")
    ax3.axhline(model.growth_agg, ls="--", lw=1.2, color=BLUE, alpha=0.5)
    ax3.axhline(model.g, ls="--", lw=1.2, color=GRN, alpha=0.5)
    ax3.axhline(0.0, ls="--", lw=1.2, color=ORG, alpha=0.5)
    stylise(ax3, "Output Growth Rates")
    ax3.legend(fontsize=7, framealpha=0.9)

    # (4) Levels
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(t_full, Y_full, lw=2.2, color=BLUE, label="Y (output)")
    ax4.plot(t_full, C_full, lw=2.2, color=GRN, label="C (consumption)")
    ax4.plot(t_full, I_full, lw=2.2, color=ORG, label="I (investment)")
    stylise(ax4, "Aggregate Levels (C, Y, I)")
    ax4.legend(fontsize=8, framealpha=0.9)

    # (5) Capital stock
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(t_full, K_full, lw=2.2, color=PURP, label="K (solved)")
    K_bgp_path = np.array([model.bgp_values((1.0 + model.g) ** t, (1.0 + model.n) ** t)[0] for t in t_full])
    ax5.plot(t_full, K_bgp_path, lw=1.5, color=GRAY, ls="--", alpha=0.7, label="BGP trajectory")
    ax5.fill_between(t_full, K_full, K_bgp_path, alpha=0.12, color=PURP, label="Gap to BGP")
    stylise(ax5, "Capital Stock vs. BGP Trajectory")
    ax5.legend(fontsize=8, framealpha=0.9)

    if out_path is not None:
        out_path = str(out_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Plot saved → {out_path}")

    plt.close(fig)


# -----------------------------------------------------------------------------
# Main program
# -----------------------------------------------------------------------------


def main():


    # Parameters
    alpha = 0.3
    delta = 0.1
    beta = 0.99
    n = 0.01
    g = 0.02
    T = 30

    print(f"JAX version : {jax.__version__}")
    print(f"Backend     : {jax.default_backend()}")
    print(f"Devices     : {jax.devices()}\n")

    # Balanced growth path setup
    A0, L0 = 1.0, 1.0

    tmp_model = RCK_model(
        alpha=alpha,
        delta=delta,
        beta=beta,
        n=n,
        g=g,
        T=T,
        A_path=jnp.asarray((1.0 + g) ** np.arange(1, T + 1), dtype=jnp.float64),
        L_path=jnp.asarray((1.0 + n) ** np.arange(1, T + 1), dtype=jnp.float64),
        K0=0.0,
        CTp1=0.0,
        YTp1=0.0,
    )

    K0_bgp, Y0_bgp, C0_bgp, I0_bgp = tmp_model.bgp_values(A0, L0)
    K0 = 0.9 * K0_bgp

    ATp1 = (1.0 + g) ** (T + 1)
    LTp1 = (1.0 + n) ** (T + 1)
    KTp1, YTp1, CTp1, ITp1 = tmp_model.bgp_values(ATp1, LTp1)

    model = RCK_model(
        alpha=alpha,
        delta=delta,
        beta=beta,
        n=n,
        g=g,
        T=T,
        A_path=jnp.asarray((1.0 + g) ** np.arange(1, T + 1), dtype=jnp.float64),
        L_path=jnp.asarray((1.0 + n) ** np.arange(1, T + 1), dtype=jnp.float64),
        K0=K0,
        CTp1=CTp1,
        YTp1=YTp1,
    )

    print("=== BGP summary ===")
    print(f"  k*  (intensive)   = {model.k_star():.6f}")
    print(f"  K0 (BGP)          = {K0_bgp:.6f}")
    print(f"  K0 (initial, 90%) = {K0:.6f}")
    print(f"  C0 (BGP)          = {C0_bgp:.6f}")
    print(f"  Y0 (BGP)          = {Y0_bgp:.6f}")
    print(f"  BGP growth rates  : agg={model.growth_agg:.4f}  per-capita={g:.4f}  intensive=0\n")

    # Initial guess
    X0 = build_initial_guess(model)

    # Solve
    print("Compiling residual and block-Jacobian paths...\n")
    t0 = time.perf_counter()
    X_sol = model.solve(X0, tol=1e-10, max_iter=60, verbose=True)
    elapsed = time.perf_counter() - t0
    print(f"\nSolve time : {elapsed:.4f} s")

    # Diagnostics
    F_sol = np.asarray(model.residual_blocks(X_sol))
    max_resid = float(np.max(np.abs(F_sol)))
    print("\n=== Convergence check ===")
    print(f"  max |residual| = {max_resid:.4e}")

    t_full, C_full, K_full, Y_full, I_full, A_full, L_full = unpack_solution(model, X_sol)

    gK_agg = growth_rate(K_full)
    gK_pc = growth_rate(K_full, L_full)
    gK_int = growth_rate(K_full, A_full * L_full)
    gY_agg = growth_rate(Y_full)
    gY_pc = growth_rate(Y_full, L_full)
    gY_int = growth_rate(Y_full, A_full * L_full)

    print(f"  gK_agg  (last period): {gK_agg[-1]:.6f}  (BGP: {model.growth_agg:.6f})")
    print(f"  gK_pc   (last period): {gK_pc[-1]:.6f}   (BGP: {model.g:.6f})")
    print(f"  gK_int  (last period): {gK_int[-1]:.6f}   (BGP: 0)")
    print(f"  gY_agg  (last period): {gY_agg[-1]:.6f}  (BGP: {model.growth_agg:.6f})")
    print(f"  gY_pc   (last period): {gY_pc[-1]:.6f}   (BGP: {model.g:.6f})")
    print(f"  gY_int  (last period): {gY_int[-1]:.6f}   (BGP: 0)")

    # Plot
    output_path = Path(__file__).resolve().parent / "figs/rck"
    make_plot(model, t_full, C_full, K_full, Y_full, I_full, A_full, L_full, out_path=output_path)
    print(f"\nPlot file: {output_path}")


if __name__ == "__main__":
    main()