from src.evolutionary import (
    evolution,
    create_data,
    initialise_agents,
    compute_all_errors,
    compute_child_coefs,
)
from src.visuals import visualise_evolution

if __name__ == "__main__":
    # Input
    polynomial = "y ~ x+x^2+x^3"  # only 2d for now
    use_bias = True

    config = {
        "N_initial_population": 200,
        "N_iterations": 100,
        "mutation_coefficient": 0.4,
        "fertility_rate": 3,
    }
    config["polynomial"] = polynomial
    config["use_bias"] = use_bias
    config["n_degree"] = config["polynomial"].count("x")
    config["n_coefs"] = (
        config["n_degree"] + 1 if config["use_bias"] else config["n_degree"]
    )
    x, y, best_coefs = evolution(**config)

    visualise_evolution(x, y, best_coefs, use_bias)


# TODO:
#
# Final logic:
#     - Take input: "y ~ x+ x^2 ... x^n" and data points
#         - Read this and determine how many coefficients are needed
#     - Refactor error computation to compute error for any polynomial
# - Add possibility to add points into a scatter plot. Read these as data. Make this fast and neat
#     - Take these point as inputs, override create_data
#         - Ignore coefficients
# Visuals docstrings and type annotations
# All docstrings update
# unittests for all components
# If no bias and only one degree = Straight line and everything breaks
# Add precommit hooks
# Profile code
# Optimise code
# If one coefficient close to zero, remove it? Or see what powers are given in the original input formula?
# Add into CV/github.io
# Deploy in Azure free tier app service
# Add github workflows (Auto deploy when new stuff)
# Make UI fancy and add explanations
#
