functions {
  real normal_lub_rng(real mu, real sigma, real lb, real ub) {
    real p_lb = normal_cdf(lb | mu, sigma);
    real p_ub = normal_cdf(ub | mu, sigma);
    real u = uniform_rng(p_lb, p_ub);
    real y = mu + sigma * inv_Phi(u);
    return y;
  }
}

data {
  real U;
  int<lower=0> N_censored;
  int<lower=0> N_observed;
  array[N_observed] real<upper=U> y;
}

parameters {
  real mu;
}

model {
  for (n in 1 : N_observed) {
    y[n] ~ normal(mu, 1.0) T[ , U];
  }
  target += N_censored * normal_lccdf(U | mu, 1);
}

generated quantities {
  array[N_observed + N_censored] real y_pred;

  for (n in 1:N_observed) {
    y_pred[n] = normal_lub_rng(mu, 1.0, negative_infinity(), U);
  }

  for (n in 1:N_censored) {
    y_pred[N_observed + n] = normal_lub_rng(mu, 1.0, U, positive_infinity());
  }
}