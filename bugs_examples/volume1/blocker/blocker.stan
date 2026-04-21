// Random effects logistic meta-analysis model with heavy-tailed heterogeneity

data {
    int<lower=0> N;
    array[N] int<lower=0> n_t;
    array[N] int<lower=0> r_t;
    array[N] int<lower=0> n_c;
    array[N] int<lower=0> r_c;
}

parameters {
    real d;
    real<lower=0> sigmasq_delta;
    vector[N] mu;
    vector[N] delta;
}

transformed parameters {
    real<lower=0> sigma_delta;
    sigma_delta = sqrt(sigmasq_delta);
}

model {
    // Priors
    mu ~ normal(0, sqrt(1E5));
    d ~ normal(0, 1E3);
    sigmasq_delta ~ inv_gamma(1E-3, 1E-3);

    // Random effect 
    delta ~ student_t(4, d, sigma_delta);

    // Likelihood
    r_t ~ binomial_logit(n_t, mu +delta);
    r_c ~ binomial_logit(n_t, mu);
}

generated quantities {
    real delta_new;
    delta_new = student_t_rng(4, d, sigma_delta);
}