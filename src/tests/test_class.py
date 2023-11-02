# pylint: skip-file
import sys

import numpy as np
import pytest

sys.path.append("src")
from evolutionary import Agent, Population, World

# create_data ############################################
config = {
    "N_initial_population": 50,
    "N_max_iter": 20,
    "mutation_coefficient": 0.4,
    "fertility_rate": 3,
}

config0 = config.copy()
config0["polynomial"] = "y ~ x+b"
config0["use_bias"] = True
config1 = config.copy()
config1["polynomial"] = "y ~ x"
config1["use_bias"] = False
config2 = config.copy()
config2["polynomial"] = "y ~ x+x^2+b"
config2["use_bias"] = True
config3 = config.copy()
config3["polynomial"] = "y ~ x+x^2"
config3["use_bias"] = False

data_test_world = [config0, config1, config2, config3]

# Backend


@pytest.mark.parametrize("config", data_test_world)
def test_world(config):
    world = World(name="pop1", **config)
    world.initialise_world_()
    assert world.current_iteration == 0
    assert world.population.size == config["N_initial_population"]
    world.evolve_()
    assert world.population.size != config["N_initial_population"]
    assert world.current_iteration > 0
    assert world.current_iteration <= config["N_max_iter"]
