## Parameterization mismatch (Stan vs TFP)

This is a running commentary on any significant model parameterization differences I encounter going through the process of translating the basic_estimator stan models into a particular Jax workflow. 

Relevant for `negative_binomial`;`negative_binomial2`:

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
