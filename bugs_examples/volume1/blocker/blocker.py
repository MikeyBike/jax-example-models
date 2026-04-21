import json
from pathlib import Path
import functools

import jax
import jax.numpy as jnp
import jax.random as jrd
import blackjax
import tensorflow_probability.substrates.jax as tfp
import warnings

warnings.filterwarnings("ignore")

tfd = tfp.distributions
tfb = tfp.bijectors


def main():
    # Data

    script_dir = Path(__file__).resolve().parent
    data_file = script_dir / "blocker_data.json"

    with open(data_file) as f:
        data = json.load(f)

    N = data["N"]
    n_c = jnp.array(data["nc"], dtype=jnp.float32)
    n_t = jnp.array(data["nt"], dtype=jnp.float32)
    r_c = jnp.array(data["rc"], dtype=jnp.float32)
    r_t = jnp.array(data["rt"], dtype=jnp.float32)

    # Model

    def random_effects_logistic_model(r_c, n_c, r_t, n_t):

        positive_bij = tfb.Exp()

        def log_density(position):
            d               = position["d"]
            sigmasq_unc     = position["sigmasq_delta_unc"]
            mu              = position["mu"]
            delta           = position["delta"]

            # Constrained parameters
            sigmasq_delta = positive_bij.forward(sigmasq_unc)
            sigma_delta   = jnp.sqrt(sigmasq_delta)

            log_det_sigmasq = positive_bij.forward_log_det_jacobian(sigmasq_unc, event_ndims=0)

            # Priors
            lp  = tfd.Normal(0.0, jnp.sqrt(1e5)).log_prob(mu).sum()
            lp += tfd.Normal(0.0, 1e3).log_prob(d)
            lp += tfd.InverseGamma(concentration=1e-3, scale=1e-3).log_prob(sigmasq_delta)

            # Random effects
            lp += tfd.StudentT(df=4.0, loc=d, scale=sigma_delta).log_prob(delta).sum()

            # Likelihood
            lp += tfd.Binomial(total_count=n_t, logits=mu + delta).log_prob(r_t).sum()
            lp += tfd.Binomial(total_count=n_c, logits=mu).log_prob(r_c).sum()

            # Log-abs-det Jacobian
            lp += log_det_sigmasq

            return lp

        def inv_transform(position):
            sigmasq_delta = positive_bij.forward(position["sigmasq_delta_unc"])
            sigma_delta   = jnp.sqrt(sigmasq_delta)
            return {
                "d":             position["d"],
                "sigmasq_delta": sigmasq_delta,
                "sigma_delta":   sigma_delta,
                "mu":            position["mu"],
                "delta":         position["delta"],
            }

        def generate(rng, d, sigma_delta):
            delta_new = tfd.StudentT(df=4.0, loc=d, scale=sigma_delta).sample(seed=rng)
            return {"delta_new": delta_new}

        def initialize_random(rng):
            k1, k2, k3, k4 = jrd.split(rng, 4)
            return {
                "d":                jrd.normal(k1),
                "sigmasq_delta_unc": jrd.normal(k2),
                "mu":               jrd.normal(k3, shape=(N,)),
                "delta":            jrd.normal(k4, shape=(N,)),
            }

        return log_density, inv_transform, generate, initialize_random

    log_density, inv_transform, generate, initialize_random = (
        random_effects_logistic_model(r_c, n_c, r_t, n_t)
    )

    # Random initialisation

    num_chains = 4

    seed = 1234781938712
    key = jrd.key(seed)
    init_key, nuts_key, key = jrd.split(key, 3)

    init_keys = jrd.split(init_key, num_chains)
    nuts_keys = jrd.split(nuts_key, num_chains)

    t_params_init = jax.vmap(initialize_random)(init_keys)
    print(f"{t_params_init=}")
    print(f"{jax.vmap(inv_transform)(t_params_init)=}")

    # NUTS sampler

    def random_markov_chain(key, kernel, init_state, num_draws):
        @jax.jit
        def one_step(state, key):
            state, _ = kernel(key, state)
            return state, state

        keys = jrd.split(key, num_draws)
        _, states = jax.lax.scan(one_step, init_state, keys)
        return states

    def nuts_sample_one_chain(key, log_density, init_position, num_draws):
        warmup_key, sample_key = jrd.split(key)
        warmup = blackjax.window_adaptation(blackjax.nuts, log_density)
        (state, params), _ = warmup.run(warmup_key, init_position, num_steps=num_draws)
        kernel = blackjax.nuts(log_density, **params).step
        states = random_markov_chain(sample_key, kernel, state, num_draws)
        return states.position

    # Posterior analysis

    num_draws = 1000

    t_draws = jax.vmap(
        lambda k, init_pos: nuts_sample_one_chain(k, log_density, init_pos, num_draws)
    )(nuts_keys, t_params_init)

    flat_t_draws = jax.tree.map(
        lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]),
        t_draws
    )

    draws = jax.vmap(inv_transform)(flat_t_draws)

    posterior_means = jax.tree.map(functools.partial(jnp.mean, axis=0), draws)
    posterior_stds  = jax.tree.map(functools.partial(jnp.std,  axis=0), draws)

    print(f"Posterior mean (d):             {posterior_means['d']:.4f}")
    print(f"Posterior std  (d):             {posterior_stds['d']:.4f}")
    print(f"Posterior mean (sigmasq_delta): {posterior_means['sigmasq_delta']:.4f}")
    print(f"Posterior std  (sigmasq_delta): {posterior_stds['sigmasq_delta']:.4f}")
    print(f"Posterior mean (sigma_delta):   {posterior_means['sigma_delta']:.4f}")
    print(f"Posterior std  (sigma_delta):   {posterior_stds['sigma_delta']:.4f}")
    for i in range(N):
        print(f"Posterior mean (mu[{i+1}]):    {posterior_means['mu'][i]:.4f}")
        print(f"Posterior std  (mu[{i+1}]):    {posterior_stds['mu'][i]:.4f}")
    for i in range(N):
        print(f"Posterior mean (delta[{i+1}]): {posterior_means['delta'][i]:.4f}")
        print(f"Posterior std  (delta[{i+1}]): {posterior_stds['delta'][i]:.4f}")

    # Generated quantities

    def generated_quantities(key, draws, generate_fn):
        S    = draws["d"].shape[0]
        keys = jrd.split(key, S)
        return jax.vmap(generate_fn)(keys, draws["d"], draws["sigma_delta"])

    key, gq_key = jrd.split(key)
    gq_draws    = generated_quantities(gq_key, draws, generate)

    gq_means = jax.tree.map(functools.partial(jnp.mean, axis=0), gq_draws)
    gq_stds  = jax.tree.map(functools.partial(jnp.std,  axis=0), gq_draws)
    print(f"\nPosterior mean (delta_new): {gq_means['delta_new']:.4f}")
    print(f"Posterior std  (delta_new): {gq_stds['delta_new']:.4f}")


if __name__ == "__main__":
    main()