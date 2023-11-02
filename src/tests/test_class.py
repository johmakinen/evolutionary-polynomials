# pylint: skip-file
import pytest
import sys
import numpy as np

sys.path.append("src")
from evolutionary import Agent, Population, World

# create_data ############################################
polynomial = "y ~ x+x^2+x^3"  # only 2d for now
use_bias = True

config = {
    "N_initial_population": 50,
    "N_max_iter": 20,
    "mutation_coefficient": 0.4,
    "fertility_rate": 3,
}
config["polynomial"] = polynomial
config["use_bias"] = use_bias

data_test_world = [config]

# Backend

@pytest.mark.parametrize("config", data_test_world)
def test_world(config):
    world = World(name="pop1", **config)
    world.initialise_world_()
    assert world.current_iteration == 0
    assert world.population.size == config['N_initial_population']
    world.evolve_()
    assert world.population.size != config['N_initial_population']
    assert world.current_iteration > 0
    assert world.current_iteration <= config['N_max_iter']



    
