from src.evolutionary import World
from src.visuals import visualise_evolution

if __name__ == "__main__":

    # Input
    polynomial = "y ~ x+x^2+x^3"  # only 2d for now
    use_bias = True

    config = {
        "N_initial_population": 100,
        "N_max_iter": 50,
        "mutation_coefficient": 0.4,
        "fertility_rate": 3,
    }
    config["polynomial"] = polynomial
    config["use_bias"] = use_bias

    # Backend
    world = World(name="pop1", **config)
    world.initialise_world_()
    x, y, best_coefs = world.evolve_()
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
