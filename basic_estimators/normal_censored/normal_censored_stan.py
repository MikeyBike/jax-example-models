from pathlib import Path

import cmdstanpy
import numpy as np


def main():
    script_dir = Path(__file__).resolve().parent
    stan_file = script_dir / "normal_censored.stan"
    data_file = script_dir / "normal_censored_data.json"

    model = cmdstanpy.CmdStanModel(stan_file=str(stan_file))

    fit = model.sample(
        data=str(data_file),
        chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        seed=123,
        show_console=False,
    )

    mu_draws = fit.stan_variable("mu")
    y_pred = fit.stan_variable("y_pred")

    print(f"Posterior mean (mu): {mu_draws.mean()}")
    print(f"Posterior std (mu): {mu_draws.std(ddof=1)}")

    print(f"Posterior predictive mean: {y_pred.mean(axis=0)}")
    print(f"Posterior predictive std: {y_pred.std(axis=0, ddof=1)}")

    print("\nSummary:")
    print(fit.summary())


if __name__ == "__main__":
    main()
