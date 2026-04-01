## Parameterization mismatch (Stan vs TFP)

This is a running commentary on any significant model parameterization differences I encounter going through the process of translating the basic_estimator stan models into a particular Jax workflow. 

### `negative_binomial` and `negative_binomial2`:

**Stan**  
`neg_binomial(alpha, beta)` has PMF:
$$
\binom{n + \alpha - 1}{\alpha - 1}
\left(\frac{\beta}{\beta + 1}\right)^\alpha
\left(\frac{1}{\beta + 1}\right)^n
$$

**TensorFlow Probability**  
`NegativeBinomial(total_count, probs)` has PMF:
$$
\binom{s + f - 1}{s}
p^s (1 - p)^f
$$

To match the two parameterizations:
$$
\text{total\_count} = \alpha, \quad
\text{probs} = \frac{1}{1 + \beta}
$$

In this model, Stan defines:
$$
\beta = \frac{p_{\text{success}}}{1 - p_{\text{success}}}
$$

Substituting:
$$
\text{probs} = \frac{1}{1 + \beta} = 1 - p_{\text{success}}
$$ 


### `normal_mixture_k`:

**Stan**  
`simplex[K] theta` with no explicit prior uses Stan's internal stick-breaking transform, which is deliberately constructed so that a flat prior in unconstrained space induces a $\text{Dirichlet}(1, \ldots, 1)$ (uniform) prior on the simplex.

**TFP**  
`tfb.SoftmaxCentered()` maps $\mathbb{R}^{K-1} \to \Delta^K$ via $\text{softmax}([x;\, 0])$. A flat prior in unconstrained space induces a different, non-uniform distribution on the simplex because its Jacobian has a different structure.

To match Stan's implicit prior, we add an explicit $\text{Dirichlet}(1, \ldots, 1)$ term in the JAX `log_density`:
```python
lp += tfd.Dirichlet(jnp.ones(K)).log_prob(theta)
```

This is added *after* the forward transform $\theta = f(\tilde{\theta})$, so $\theta \in \Delta^K$ is already in constrained space. The existing $\log \left|\det J_f(\tilde{\theta})\right|$ term correctly handles the change of variables — no double-counting occurs.