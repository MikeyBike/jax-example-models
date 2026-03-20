import jax
import jax.numpy as jnp
import jax.random as jrd
import functools
import blackjax
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors



def main():
    # Data

    N = 9
    y = jnp.array([0, 1, 4, 0, 2, 2, 5, 0, 1])

    
    # Model defintion
    
    def negative_binomial_model(y):
        y = jnp.asarray(y).astype(jnp.int32)
        N = y.shape[0]
        exp_bij = tfb.Exp()

        def log_density(position):
            alpha_unc = position["alpha_unconstrained"]
            beta_unc = position["beta_unconstrained"]

            alpha = exp_bij.forward(alpha_unc)
            beta = exp_bij.forward(beta_unc)

            log_det_alpha = exp_bij.forward_log_det_jacobian(alpha_unc, event_ndims=0)
            log_det_beta = exp_bij.forward_log_det_jacobian(beta_unc, event_ndims=0)

            lp = tfd.HalfCauchy(loc=0.0, scale=10.0).log_prob(alpha)
            lp += tfd.HalfCauchy(loc=0.0, scale=10.0).log_prob(beta)
            lp += jnp.sum(tfd.NegativeBinomial(total_count=alpha, probs=1.0 / (beta + 1.0)).log_prob(y))
            lp += log_det_alpha + log_det_beta
            return lp
        
        def inv_transform(position):
            return {
                "alpha": exp_bij.forward(position["alpha_unconstrained"]),
                "beta": exp_bij.forward(position["beta_unconstrained"])
            }
        
        def generate(rng, alpha, beta):
            return {
                "y_pred": tfd.NegativeBinomial(total_count=alpha, probs=1.0 / (beta + 1.0)).sample(seed=rng, sample_shape=(N,))
            }
        
        def initialize_random(rng):
            k1, k2 = jrd.split(rng, 2)
            return {
                "alpha_unconstrained": jrd.normal(k1, shape=()),
                "beta_unconstrained": jrd.normal(k2, shape=())
            }
        
        return log_density, inv_transform, generate, initialize_random
    
    
    log_density, inv_transform, generate, initialize_random = negative_binomial_model(y)
    
    # Random initialization

    seed = 1234781938712
    key = jrd.key(seed)
    init_key, nuts_key, key = jrd.split(key, 3)
    
    t_params_init = initialize_random(init_key)
    print(f"{t_params_init=}")

    params_init = inv_transform(t_params_init)
    print(f"{params_init}")

    #  NUTS sampler

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
    
    num_draws= 1000
    t_draws = nuts_sample(nuts_key, log_density, t_params_init, num_draws)
    

    # Posterior Analysis

    def inv_transform_draws(t_draws):
        return jax.vmap(inv_transform)(t_draws)
    
    draws = inv_transform_draws(t_draws)

    
    posterior_means = jax.tree.map(functools.partial(jnp.mean, axis=0), draws)
    posterior_stds = jax.tree.map(functools.partial(jnp.std, axis=0), draws)
    print(f"{posterior_means=}")
    print(f"{posterior_stds=}")

    # Posterior predictive checks

    def posterior_predictive_check(key, draws, generate_fn):
        S = draws["alpha"].shape[0]
        keys = jrd.split(key, S)
        return jax.vmap(generate_fn, in_axes=(0, 0, 0))(keys, draws["alpha"], draws["beta"])

    key, gq_key = jrd.split(key, 2)
    pred_draws = posterior_predictive_check(gq_key, draws, generate)

    posterior_predictive_means = jax.tree.map(functools.partial(jnp.mean, axis=0), pred_draws)
    posterior_predictive_stds = jax.tree.map(functools.partial(jnp.std, axis=0), pred_draws)
    print(f"{posterior_predictive_means=}")
    print(f"{posterior_predictive_stds=}")

    # Parameters vary between jax and stan because of the heavy-tailed, weakly-identified posterior, so the means are dominated by occasional large draws in the tail. Posterior predictive checks match tho. 

if __name__ == "__main__":
    main()


