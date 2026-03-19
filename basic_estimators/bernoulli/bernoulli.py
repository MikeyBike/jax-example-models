import jax
import jax.numpy as jnp
import jax.random as jrd
import functools
import blackjax
import distrax


def main():
    # Data

    N = 10
    y = jnp.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])


    # Model definition

    def bernoulli_model(y):
        y = jnp.asarray(y).astype(jnp.int32)
        N = y.shape[0]
        sigmoid_bij = distrax.Sigmoid()

        def log_density(position):
            mu = position["theta_unconstrained"]
            theta, log_det = sigmoid_bij.forward_and_log_det(mu)
            lp  = distrax.Beta(alpha=1.0, beta=1.0).log_prob(theta)
            lp += jnp.sum(distrax.Bernoulli(probs=theta).log_prob(y))
            lp += log_det
            return lp

        def inv_transform(position):
            mu = position["theta_unconstrained"]
            return {"theta": sigmoid_bij.forward(mu)}

        def generate(rng, theta):
            return {"y_pred": distrax.Bernoulli(probs=theta).sample(seed=rng, sample_shape=(N,))}

        def initialize_random(rng):
            k1, = jrd.split(rng, 1)
            return {"theta_unconstrained": jrd.normal(k1, shape=())}

        return log_density, inv_transform, generate, initialize_random

    log_density, inv_transform, generate, initialize_random = bernoulli_model(y)

    # Random initialization

    seed = 123357181
    key = jrd.key(seed)
    init_key, nuts_key, key = jrd.split(key, 3)  

    t_params_init = initialize_random(init_key)
    print(f"{t_params_init=}")

    params_init = inv_transform(t_params_init)
    print(f"{params_init=}")

    # Nuts sampler

    def random_markov_chain(key, kernel, init_state, num_draws):
        @jax.jit
        def one_step(state, key):
            state, _ = kernel(key, state)
            return state, state
        
        keys = jrd.split(key, num_draws)
        _, states = jax.lax.scan(one_step, init_state, keys)
        return states

    def nuts_sample(key, log_density, init_position, num_draws):
        init_key, warmup_key, sample_key = jrd.split(key, 3)
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
        S = draws["theta"].shape[0]
        keys = jrd.split(key, S)
        return jax.vmap(generate_fn, in_axes=(0, 0))(keys, draws["theta"])

    key, gq_key = jrd.split(key, 2)
    pred_draws = posterior_predictive_check(gq_key, draws, generate)

    posterior_predictive_means = jax.tree.map(functools.partial(jnp.mean, axis=0), pred_draws)
    posterior_predictive_stds = jax.tree.map(functools.partial(jnp.std, axis=0), pred_draws)
    print(f"{posterior_predictive_means=}")
    print(f"{posterior_predictive_stds=}")


if __name__ == "__main__":
    main()





