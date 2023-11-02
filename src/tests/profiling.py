import cProfile
import sys

sys.path.append("src")
from evolutionary import World


def main():
    polynomial = "y ~ x+x^2+x^3+x^4+b"  # only 2d for now
    use_bias = True

    config = {
        "N_initial_population": 200,
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


if __name__ == "__main__":
    cProfile.run("main()", sort="tottime")
