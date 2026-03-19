import jax
import jax.numpy as jnp
import jax.random as jrd
import functools
import blackjax
import distrax
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors



def main():
    # Data

    N = 9
    y = jnp.array([0, 1, 4, 0, 2, 2, 5, 0, 1])

    
    # Model defintion
    
    def negative_binomial_model(y):
        y = jnp.asarray(y).astyep(jnp.int32)
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
            lp += jnp.sum(tfd.NegativeBinomial(total_counts=alpha, probs=beta / (beta + 1.0)).log_prob(y))
            lp += log_det_alpha + log_det_beta
            return lp
        
        def inv_transform(position):
            return {
                "alpha": exp_bij.forward(position["alpha_unconstrained"]),
                "beta": exp_bij.forward(position["beta_unconstrained"])
            }
        
        def generate(rng, alpha, beta):
            return {
                "y_pred": tfd.NegativeBinomial(total_counts=alpha, probs=beta / (beta + 1.0)).sample(seed=rng, sample_shape=(N,))
            }
        
        def initialize_random(rng):
            k1, k2 = jrd.split(rng, 2)
            return {
                "alpha_unconstrained": jrd.normal(k1, shape=()),
                "beta_unconstrained": jrd.normal(k2, shape=())
            }
        
        return log_density, inv_transform, generate, initialize_random
    
        



if __name__ == "__main__":
    main()


