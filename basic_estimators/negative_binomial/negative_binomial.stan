data {
    int<lower=1> N;
}

parameters {
    real<lower=0> alpha;
    real<lower=0> beta;
}

model {
    alpha ~ cauchy(0,10);
    beta ~ cauchy(0,10);
    y ~ neg_binomial(alpha, beta);
}