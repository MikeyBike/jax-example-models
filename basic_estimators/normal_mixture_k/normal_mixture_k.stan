data {
  int<lower=1> K;
  int<lower=1> N;
  vector[N] y;
}

parameters {
  simplex[K] theta;
  vector[K] mu;
  vector<lower=0, upper=10>[K] sigma;
}

model {
  array[N] vector[K] ps;
  mu ~ normal(0, 10);
  for (n in 1 : N) {
    for (k in 1 : K) {
      ps[n][k] = normal_lupdf(y[n] | mu[k], sigma[k]);
    }
  }
  target += log_mix(theta, ps);
}

generated quantities {
  vector[N] y_pred;
  for (n in 1:N) {
    int k = categorical_rng(theta);
    y_pred[n] = normal_rng(mu[k], sigma[k]);
  }
}