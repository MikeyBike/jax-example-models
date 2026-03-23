from pathlib import Path

import cmdstanpy
import numpy as np


def main():
    N = 9
    y = np.array([0, 1, 4, 0, 2, 2, 5, 0, 1], dtype=int)

    data = {
        "N": N,
        "y": y,
    }

    script_dir = Path(__file__).resolve().parent
    stan_file = script_dir / "negative_binomial2.stan"

    model = cmdstanpy.CmdStanModel(stan_file=str(stan_file))

    fit = model.sample(
        data=data,
        chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        seed=123,
        show_console=False,
    )

    alpha_draws = fit.stan_variable("alpha")
    p_success_draws = fit.stan_variable("p_success")
    y_pred = fit.stan_variable("y_pred")

    print(f"Posterior mean (alpha): {alpha_draws.mean()}")
    print(f"Posterior std (alpha): {alpha_draws.std(ddof=1)}")

    print(f"Posterior mean (p_success): {p_success_draws.mean()}")
    print(f"Posterior std (p_success): {p_success_draws.std(ddof=1)}")

    print(f"Posterior predictive mean: {y_pred.mean(axis=0)}")
    print(f"Posterior predictive std: {y_pred.std(axis=0, ddof=1)}")

    print("\nSummary:")
    print(fit.summary())


if __name__ == "__main__":
    main()