# pylint: skip-file
import pytest
import sys
import numpy as np
from itertools import combinations

sys.path.append("src")
from evolutionary import (
    create_data,
    initialise_agents,
    compute_error,
    compute_child_coefs,
    pair_and_cull,
    Agent,
)

# create_data ############################################
data_create_data = [
    (True, 1, 10),
    (True, 3, 20),
    (False, 1, 10),
    (False, 3, 20),
    (True, 0, 10),
    (False, 0, 10),
] 


@pytest.mark.parametrize(
    "bias,degree,N", data_create_data
)  # ,"5","6","7","8"
def test_create_data(bias, degree, N):
    x, y = create_data(bias, degree, N)
    assert len(x) == len(y)
    assert len(x) == N
    assert max(x) != min(x)
    if (degree == 0) and (N > 0):
        assert max(y) == min(y)
    elif (N > 0):
        assert max(y) != min(y)
    else:
        assert np.array_equal(x,np.array([]))
        assert np.array_equal(y,np.array([]))

# initialise_agents ############################################
data_initialise_agents = [(1, 1), (3, 5), (5, 6)]  # Future: ,(0,1),(1,0)

@pytest.mark.parametrize("n_coefs,n", data_initialise_agents)
def test_initialise_agents(n_coefs, n):
    agents = initialise_agents(n_coefs, n)

    # All agents have different initial coefs
    for pair in combinations(agents, 2):
        assert all(pair[0].coef != pair[1].coef)

    # Correct number of coefs
    assert all([len(agent.coef) == n_coefs for agent in agents])

    # correct number of agents
    assert len(agents) == n


# compute_error ############################################
x = np.arange(0, 10)
data_compute_error = [
    (x, 3 * x, Agent(name="perfect", coef=[3]), False, 0),
    (x, 3 * x + 15, Agent(name="perfect_bias", coef=[15, 3]), True, 0),
    (x, x + 3 * x**2, Agent(name="perfect_poly", coef=[1, 3]), False, 0),
    (x, x + 3 * x**2 + 15, Agent(name="perfect_poly_bias", coef=[15, 1, 3]), True, 0),
    (x, x + 3 * x**2 + 15, Agent(name="poly_bias_error_1", coef=[16, 1, 3]), True, 1),
]


@pytest.mark.parametrize(
    "x,y,agent,use_bias,expected",
    data_compute_error,
    ids=[
        "perfect",
        "perfect_bias",
        "perfect_poly",
        "perfect_poly_bias",
        "poly_bias_error_1",
    ],
)
def test_compute_error(x, y, agent, use_bias, expected):
    assert compute_error(x, y, agent, use_bias) == expected


# compute_child_coefs ############################################
data_compute_child_coefs = [
    (
        Agent(name="p1", coef=np.array([1, 2]), error=10),
        Agent(name="p2", coef=np.array([3, 1]), error=20),
        0,
        np.array([1.0, 2.0]) * (2 / 3) + np.array([3.0, 1.0]) * (1 / 3),
    )
]


@pytest.mark.parametrize(
    "p1,p2,mutation_coefficient,expected", data_compute_child_coefs
)
def test_compute_child_coefs(p1, p2, mutation_coefficient, expected):
    W = np.array([p1.error, p2.error])
    W = 1 - W / W.sum()
    res = compute_child_coefs(p1, p2, mutation_coefficient)
    assert np.allclose(res, expected)  # Floating points can differ
    assert np.sum(W) == 1


@pytest.mark.parametrize(
    "x,y,agent,use_bias,expected",
    data_compute_error,
    ids=[
        "perfect",
        "perfect_bias",
        "perfect_poly",
        "perfect_poly_bias",
        "poly_bias_error_1",
    ],
)
def test_compute_error(x, y, agent, use_bias, expected):
    assert compute_error(x, y, agent, use_bias) == expected


# pair_and_cull ############################################
data_pair_and_cull = [
    (
        [
            Agent(name="1", coef=[1, 1], error=1),
            Agent(name="2", coef=[1, 1], error=2),
            Agent(name="3", coef=[1, 1], error=3),
            Agent(name="4", coef=[1, 1], error=4),
            Agent(name="5", coef=[1, 1], error=5),
            Agent(name="6", coef=[1, 1], error=6),
            Agent(name="7", coef=[1, 1], error=7),
            Agent(name="8", coef=[1, 1], error=8),
            Agent(name="9", coef=[1, 1], error=9),
            Agent(name="10", coef=[1, 1], error=10),
        ],
        0.8,
        4,
    )
]


@pytest.mark.parametrize("agents,survivability,expected", data_pair_and_cull)
def test_pair_and_cull(agents, survivability, expected):
    current_best_agent, paired_agents = pair_and_cull(agents, survivability)

    assert isinstance(current_best_agent, Agent)
    assert paired_agents[0][0].error + paired_agents[0][1].error == 3
    assert paired_agents[3][0].error + paired_agents[3][1].error == 15
    assert len(paired_agents) == expected


#####
# pytest src/tests/test_utils.py -v # This file
# pytest src/tests/ -v # All tests
