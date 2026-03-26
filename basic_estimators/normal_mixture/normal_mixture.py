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
    data_file = script_dir / "normal_mixture_data.json"

    with open(data_file) as f:
        data = json.load(f)

    N = data["N"]
    y = jnp.array(data["y"])

    # Model definition

    def normal_mixture_model(y):
        y = jnp.asarray(y)
        N = y.shape[0]

        def log_density(position):
            theta_unc = position["theta_unc"]   
            mu_unc = position["mu_unc"]         

            theta = jax.nn.sigmoid(theta_unc)
            mu1 = mu_unc[0]
            mu2 = mu1 + jnp.exp(mu_unc[1])      

            log_det = (jnp.log(theta) + jnp.log1p(-theta)  + mu_unc[1])

            lp = tfd.Uniform(0.0, 1.0).log_prob(theta)
            lp += tfd.Normal(0.0, 10.0).log_prob(mu1)
            lp += tfd.Normal(0.0, 10.0).log_prob(mu2)

            log_p1 = jnp.log(theta)       + tfd.Normal(mu1, 1.0).log_prob(y)
            log_p2 = jnp.log1p(-theta)    + tfd.Normal(mu2, 1.0).log_prob(y)
            lp += jnp.sum(jnp.logaddexp(log_p1, log_p2))

            lp += log_det

            return lp

        def inv_transform(position):
            theta_unc = position["theta_unc"]
            mu_unc = position["mu_unc"]
            theta = jax.nn.sigmoid(theta_unc)
            mu1 = mu_unc[0]
            mu2 = mu1 + jnp.exp(mu_unc[1])
            return {"theta": theta, "mu": jnp.stack([mu1, mu2])}

        def generate(rng, theta, mu):
            k1, k2a, k2b = jrd.split(rng, 3)
            component = jrd.bernoulli(k1, theta, shape=(N,))
            y_comp1 = tfd.Normal(mu[0], 1.0).sample(seed=k2a, sample_shape=(N,))
            y_comp2 = tfd.Normal(mu[1], 1.0).sample(seed=k2b, sample_shape=(N,))
            return {"y_pred": jnp.where(component, y_comp1, y_comp2)}

        def initialize_random(rng):
            k1, k2 = jrd.split(rng)
            return {
                "theta_unc": jrd.normal(k1, shape=()),
                "mu_unc": jrd.normal(k2, shape=(2,)),
            }

        return log_density, inv_transform, generate, initialize_random

    log_density, inv_transform, generate, initialize_random = normal_mixture_model(y)

    # Random initialization

    seed = 1234781938712
    key = jrd.key(seed)
    init_key, nuts_key, key = jrd.split(key, 3)

    t_params_init = initialize_random(init_key)
    print(f"{t_params_init=}")

    params_init = inv_transform(t_params_init)
    print(f"{params_init=}")

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

    print(f"Posterior mean (theta): {posterior_means['theta']:.4f}")
    print(f"Posterior std  (theta): {posterior_stds['theta']:.4f}")
    for i, label in enumerate(["lower", "upper"]):
        print(f"Posterior mean  (mu[{label}]): {posterior_means['mu'][i]:.4f}")
        print(f"Posterior std   (mu[{label}]): {posterior_stds['mu'][i]:.4f}")

    # Posterior predictive checks

    def posterior_predictive_check(key, draws, generate_fn):
        S = draws["theta"].shape[0]
        keys = jrd.split(key, S)
        return jax.vmap(generate_fn)(keys, draws["theta"], draws["mu"])

    key, gq_key = jrd.split(key)
    pred_draws = posterior_predictive_check(gq_key, draws, generate)

    posterior_predictive_means = jax.tree.map(
        functools.partial(jnp.mean, axis=0), pred_draws
    )
    posterior_predictive_stds = jax.tree.map(
        functools.partial(jnp.std, axis=0), pred_draws
    )
    print(f"Posterior predictive mean: {posterior_predictive_means['y_pred']}")
    print(f"Posterior predictive std:  {posterior_predictive_stds['y_pred']}")


if __name__ == "__main__":
    main()