from pathlib import Path

import cmdstanpy
import numpy as np


def main():
    script_dir = Path(__file__).resolve().parent
    stan_file = script_dir / "normal_mixture_k.stan"
    data_file = script_dir / "normal_mixture_k_data.json"

    model = cmdstanpy.CmdStanModel(stan_file=str(stan_file))

    fit = model.sample(
        data=str(data_file),
        chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        seed=123,
        show_console=False,
    )

    theta_draws = fit.stan_variable("theta")
    mu_draws = fit.stan_variable("mu")
    sigma_draws = fit.stan_variable("sigma")
    y_pred = fit.stan_variable("y_pred")

    for k in range(theta_draws.shape[1]):
        print(f"Posterior mean (theta[{k+1}]): {theta_draws[:, k].mean():.4f}")
        print(f"Posterior std  (theta[{k+1}]): {theta_draws[:, k].std(ddof=1):.4f}")

    for k in range(mu_draws.shape[1]):
        print(f"Posterior mean (mu[{k+1}]):    {mu_draws[:, k].mean():.4f}")
        print(f"Posterior std  (mu[{k+1}]):    {mu_draws[:, k].std(ddof=1):.4f}")

    for k in range(sigma_draws.shape[1]):
        print(f"Posterior mean (sigma[{k+1}]): {sigma_draws[:, k].mean():.4f}")
        print(f"Posterior std  (sigma[{k+1}]): {sigma_draws[:, k].std(ddof=1):.4f}")

    print(f"\nPosterior predictive mean: {y_pred.mean(axis=0)}")
    print(f"Posterior predictive std:  {y_pred.std(axis=0, ddof=1)}")

    print("\nSummary:")
    print(fit.summary())


if __name__ == "__main__":
    main()
