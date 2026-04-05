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
    data_file = script_dir / "normal_mixture_k_prop_data.json"

    with open(data_file) as f:
        data = json.load(f)

    K = data["K"]
    N = data["N"]
    y = jnp.array(data["y"])

    # Model

    def normal_mixture_k_prop_model(y, K):

        simplex_bij  = tfb.SoftmaxCentered()
        mu_scale_bij = tfb.Softplus()
        sigma_bij    = tfb.Softplus()

        def log_density(position):
            theta_unc    = position["theta_unc"]
            mu_prop_unc  = position["mu_prop_unc"]
            mu_loc       = position["mu_loc"]
            mu_scale_unc = position["mu_scale_unc"]
            sigma_unc    = position["sigma_unc"]

            theta    = simplex_bij.forward(theta_unc)
            mu_prop  = simplex_bij.forward(mu_prop_unc)
            mu_scale = mu_scale_bij.forward(mu_scale_unc)
            sigma    = sigma_bij.forward(sigma_unc)

            mu = mu_loc + mu_scale * jnp.cumsum(mu_prop)

            log_det_theta    = simplex_bij.forward_log_det_jacobian(theta_unc, event_ndims=1)
            log_det_mu_prop  = simplex_bij.forward_log_det_jacobian(mu_prop_unc, event_ndims=1)
            log_det_mu_scale = mu_scale_bij.forward_log_det_jacobian(mu_scale_unc, event_ndims=0)
            log_det_sigma    = sigma_bij.forward_log_det_jacobian(sigma_unc, event_ndims=1)

           
            lp  = tfd.Dirichlet(jnp.ones(K)).log_prob(theta)
            lp += tfd.Dirichlet(jnp.ones(K)).log_prob(mu_prop)
            lp += tfd.Cauchy(0.0, 5.0).log_prob(mu_loc)
            lp += tfd.HalfCauchy(0.0, 5.0).log_prob(mu_scale)
            lp += tfd.HalfCauchy(0.0, 5.0).log_prob(sigma).sum()

            
            component_logps = (
                jnp.log(theta)[None, :]
                + tfd.Normal(mu[None, :], sigma[None, :]).log_prob(y[:, None])
            )
            lp += jax.scipy.special.logsumexp(component_logps, axis=1).sum()

            lp += log_det_theta + log_det_mu_prop + log_det_mu_scale + log_det_sigma

            return lp

        def inv_transform(position):
            theta    = simplex_bij.forward(position["theta_unc"])
            mu_prop  = simplex_bij.forward(position["mu_prop_unc"])
            mu_loc   = position["mu_loc"]
            mu_scale = mu_scale_bij.forward(position["mu_scale_unc"])
            sigma    = sigma_bij.forward(position["sigma_unc"])
            mu       = mu_loc + mu_scale * jnp.cumsum(mu_prop)

            return {
                "theta":    theta,
                "mu_prop":  mu_prop,
                "mu_loc":   mu_loc,
                "mu_scale": mu_scale,
                "sigma":    sigma,
                "mu":       mu,
            }

        def generate(rng, theta, mu, sigma):
            k1, k2 = jrd.split(rng)
            components = tfd.Categorical(probs=theta).sample(seed=k1, sample_shape=(len(y),))
            y_pred = tfd.Normal(mu[components], sigma[components]).sample(seed=k2)
            return {"y_pred": y_pred}

        def initialize_random(rng):
            k1, k2, k3, k4, k5 = jrd.split(rng, 5)
            return {
                "theta_unc":    jrd.normal(k1, shape=(K - 1,)),
                "mu_prop_unc":  jrd.normal(k2, shape=(K - 1,)),
                "mu_loc":       jrd.normal(k3),
                "mu_scale_unc": jrd.normal(k4),
                "sigma_unc":    jrd.normal(k5, shape=(K,)),
            }

        return log_density, inv_transform, generate, initialize_random

    log_density, inv_transform, generate, initialize_random = normal_mixture_k_prop_model(y, K)

    # Random initialization

    seed = 1234781938712
    key = jrd.key(seed)
    init_key, nuts_key, key = jrd.split(key, 3)

    t_params_init = initialize_random(init_key)
    print(f"{t_params_init=}")
    print(f"{inv_transform(t_params_init)=}")

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

    # Posterior analysis

    num_draws = 1000
    t_draws = nuts_sample(nuts_key, log_density, t_params_init, num_draws)

    draws = jax.vmap(inv_transform)(t_draws)

    posterior_means = jax.tree.map(functools.partial(jnp.mean, axis=0), draws)
    posterior_stds  = jax.tree.map(functools.partial(jnp.std,  axis=0), draws)

    for k in range(K):
        print(f"Posterior mean (theta[{k+1}]): {posterior_means['theta'][k]:.4f}")
        print(f"Posterior std  (theta[{k+1}]): {posterior_stds['theta'][k]:.4f}")
    for k in range(K):
        print(f"Posterior mean (mu[{k+1}]):    {posterior_means['mu'][k]:.4f}")
        print(f"Posterior std  (mu[{k+1}]):    {posterior_stds['mu'][k]:.4f}")
    for k in range(K):
        print(f"Posterior mean (sigma[{k+1}]): {posterior_means['sigma'][k]:.4f}")
        print(f"Posterior std  (sigma[{k+1}]): {posterior_stds['sigma'][k]:.4f}")

    print(f"\nPosterior mean (mu_loc):   {posterior_means['mu_loc']:.4f}")
    print(f"Posterior std  (mu_loc):   {posterior_stds['mu_loc']:.4f}")
    print(f"Posterior mean (mu_scale): {posterior_means['mu_scale']:.4f}")
    print(f"Posterior std  (mu_scale): {posterior_stds['mu_scale']:.4f}")

    # Posterior predictive check

    def posterior_predictive_check(key, draws, generate_fn):
        S = draws["theta"].shape[0]
        keys = jrd.split(key, S)
        return jax.vmap(generate_fn)(keys, draws["theta"], draws["mu"], draws["sigma"])

    key, gq_key = jrd.split(key)
    pred_draws = posterior_predictive_check(gq_key, draws, generate)

    posterior_predictive_means = jax.tree.map(functools.partial(jnp.mean, axis=0), pred_draws)
    posterior_predictive_stds  = jax.tree.map(functools.partial(jnp.std,  axis=0), pred_draws)
    print(f"\nPosterior predictive mean: {jnp.mean(posterior_predictive_means['y_pred']):.4f}")
    print(f"Posterior predictive std:  {jnp.mean(posterior_predictive_stds['y_pred']):.4f}")


if __name__ == "__main__":
    main()