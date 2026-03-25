import json
from pathlib import Path
import functools

import jax 
import jax.numpy as jnp
import jax.random as jrd
import blackjax
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors



def main():
    # Data

    script_dir = Path(__file__).resolve().parent
    data_file = script_dir / "normal_censored_data.json"

    with open(data_file) as f:
        data = json.load(f)

    U = data["U"]
    N_censored = data["N_censored"]
    N_observed = data["N_observed"]
    y = jnp.array(data["y"])

    # Model definition

    def normal_censored_model(y, U, N_censored):
        y = jnp.asarray(y)
        N_observed = y.shape[0]

        def log_density(position):
            mu = position["mu"]

            # Observed truncated normal
            log_normalized = tfd.Normal(mu, 1.0).log_cdf(U)
            lp = jnp.sum(tfd.Normal(mu, 1.0).log_prob(y) - log_normalized)

            # Censored 
            lp += N_censored * tfd.Normal(mu, 1.0).log_survival_function(U)

            return lp
        
        def inv_transform(position):
            return {"mu": position["mu"]}

        def generate(rng, mu):
            k1, k2 = jrd.split(rng)

            # Observed
            u_obs = jrd.uniform(k1, shape=(N_observed), minval=0.0, maxval=1.0)
            p_obs = tfd.Normal(mu, 1.0).cdf(U)
            y_obs = tfd.Normal(mu, 1.0).quantile(u_obs * p_obs)

            # Censored
            u_cens = jrd.uniform(k2, shape=(N_censored), minval=0.0, maxval=1.0)
            p_lb = tfd.Normal(mu, 1.0).cdf(U)
            y_cens = tfd.Normal(mu, 1.0).quantile(p_lb + u_cens * (1.0 - p_lb))

            return {"y_pred": jnp.concatenate([y_obs, y_cens])}
        

        def initialize_random(rng):
            return {"mu": jrd.normal(rng, shape=())}

        return log_density, inv_transform, generate, initialize_random

    log_density, inv_transform, generate, initialize_random = normal_censored_model(y, U, N_censored)

    # Random initialization

    seed = 1234781938712
    key = jrd.key(seed)
    init_key, nuts_key, key = jrd.split(key, 3)

    t_params_init = initialize_random(init_key)
    print(f"{t_params_init=}")

    params_init = inv_transform(t_params_init)
    print(f"{params_init}")

    # NUTS sampler

    def random_markov_chain(key, kernel, init_state, num_draws):
        @jax.jit
        def one_step(state, key):
            state, _ = kernel(key, state)
            return state, state

        keys = jrd.split(key, num_draws)
        _, states = jax.lax.scan(one_step, init_state, keys)
        return states

    def nuts_sample(key, log_density, init_position, num_draws):
        _, warmup_key, sample_key = jrd.split(key, 3)
        warmup = blackjax.window_adaptation(blackjax.nuts, log_density)
        (state, params), _ = warmup.run(warmup_key, init_position, num_steps=num_draws)
        kernel = blackjax.nuts(log_density, **params).step
        states = random_markov_chain(sample_key, kernel, state, num_draws)
        return states.position

    num_draws = 1000
    t_draws = nuts_sample(nuts_key, log_density, t_params_init, num_draws)

    # Posterior analysis

    def inv_transform_draws(t_draws):
        return jax.vmap(inv_transform)(t_draws)

    draws = inv_transform_draws(t_draws)

    posterior_means = jax.tree.map(functools.partial(jnp.mean, axis=0), draws)
    posterior_stds = jax.tree.map(functools.partial(jnp.std, axis=0), draws)
    print(f"{posterior_means=}")
    print(f"{posterior_stds=}")

    # Posterior predictive checks

    def posterior_predictive_check(key, draws, generate_fn):
        S = draws["mu"].shape[0]
        keys = jrd.split(key, S)
        return jax.vmap(generate_fn, in_axes=(0, 0))(keys, draws["mu"])

    key, gq_key = jrd.split(key, 2)
    pred_draws = posterior_predictive_check(gq_key, draws, generate)

    posterior_predictive_means = jax.tree.map(
        functools.partial(jnp.mean, axis=0), pred_draws
    )
    posterior_predictive_stds = jax.tree.map(
        functools.partial(jnp.std, axis=0), pred_draws
    )
    print(f"{posterior_predictive_means=}")
    print(f"{posterior_predictive_stds=}")


if __name__ == "__main__":
    main()