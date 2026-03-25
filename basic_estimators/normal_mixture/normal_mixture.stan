data {
  int<lower=0> N;
  vector[N] y;
}

parameters {
  real<lower=0, upper=1> theta;
  ordered[2] mu;           
}

model {
  theta ~ uniform(0, 1);   
  mu ~ normal(0, 10);
  for (n in 1 : N) {
    target += log_mix(theta, normal_lupdf(y[n] | mu[1], 1),
                             normal_lupdf(y[n] | mu[2], 1));
  }
}

generated quantities {
  vector[N] y_pred;
  for (n in 1:N) {
    y_pred[n] = bernoulli_rng(theta) ? normal_rng(mu[1], 1) 
                                     : normal_rng(mu[2], 1);
  }
}
