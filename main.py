from src.evolutionary import World
from src.visuals import visualise_evolution

if __name__ == "__main__":
    # Input
    polynomial = "y ~ x+x^2+b"  # only 2d for now
    use_bias = True

    config = {
        "N_initial_population": 100,
        "N_max_iter": 50,
        "mutation_coefficient": 0.5,
        "fertility_rate": 3,
    }
    config["polynomial"] = polynomial
    config["use_bias"] = use_bias

    # Backend
    world = World(name="pop1", **config)
    world.initialise_world_()
    x, y, best_coefs = world.evolve_()
    # visualise_evolution(x, y, best_coefs, use_bias)


# TODO:
#
# Make user able to put points on a plot
#   Take these points as x and y
#   Then, user inputs the polynomial degree, and whether to use bias term
#   -> Evolve
# Deploy in Azure free tier app service
# Make GUI usable
# Add into CV/github.io
# Add github workflows (pytests + Auto deploy when new stuff)
# Make UI fancy and add explanations
#
