# Imports
import random
import string
from dataclasses import dataclass, field
from itertools import pairwise
import os
from typing import Iterable, Union
import operator
import logging
import numpy as np


def init_logger(
    level_: int = logging.INFO, log_file: str = "genetic_LR.log"
) -> logging.Logger:
    """Initialised logger for the chosen level.

    Args:
        level_ (int, optional): Level to log. Defaults to logging.INFO.
        log_file (str, optional): Logs are saved here. Defaults to "genetic_LR.log".

    Returns:
        logging.Logger: Current logger
    """
    os.remove(log_file)

    logger_ = logging.getLogger(__name__)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)
    logger_.setLevel(level_)  # <<< Added Line
    c_handler.setLevel(level_)
    f_handler.setLevel(level_)

    # Create formatters and add it to handlers
    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger_
    logger_.addHandler(c_handler)
    logger_.addHandler(f_handler)
    return logger_


def seed_all(seed: int = 42) -> None:
    """Seeds random and np.random

    Args:
        seed (int, optional): Seed. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Seeded everything with seed %s", seed)


@dataclass(order=True)
class Agent:
    """Dataclass for Agents (workers)

    Each agent has:
        - Name
        - Coefficients in a list
        - Error from using the coefficients
    """

    name: str
    coef: list  # [b,x0,x1,x2,...]
    error: float = np.inf
    sort_index: int = field(init=False)

    def __post_init__(self):
        self.sort_index = self.error


# Create data
def create_data(N: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Creates the data from a given polynomial.

    Args:
        N (int, optional): How many data points are wanted. Defaults to 100.

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    x = np.arange(0, N)
    k = 3.14
    b = 7
    y = k * x + b
    y += np.random.normal(scale=50, size=y.shape)
    logger.info(
        "Created %d datapoints with coefficients with y = %.2fx + %.2f", N, k, b
    )
    return x, y


def id_generator(
    size: int = 20, chars: str = string.ascii_uppercase + string.digits
) -> str:
    return "".join(random.choice(chars) for _ in range(size))


# Create 10 agents
def initialise_agents(n_coefs: int, n: int = 10) -> list[Agent]:
    """Generates the first generation agents.

    Args:
        n_coefs (int): How many coefficients there are
        n (int, optional): How many agents to create initially. Defaults to 10.

    Returns:
        list[Agent]: List of initial agents
    """
    agents = [
        Agent(name=id_generator(), coef=np.random.randint(-10, 10, size=n_coefs))
        for i in range(n)
    ]
    logger.info("Created %d initial agents", n)
    return agents


# Compute error of an agent
def compute_error(
    x: Iterable[Union[float, int]], y: Iterable[Union[float, int]], agent: Agent
) -> None:
    """Computed the error for a single agent. Currently uses simple ordinary least squares

    Args:
        x (Iterable[Union[float,int]]): x values
        y (Iterable[Union[float,int]]): y values
        agent (Agent)
    """
    agent.error = np.sum((y - (agent.coef[0] + agent.coef[1] * x)) ** 2)
    agent.__post_init__()


def compute_all_errors(x, y, agents: list[Agent]):
    for agent in agents:
        compute_error(x, y, agent)


# Pair agents, Kill 10% worst pairs to limit resources
def pair_and_cull(
    agents: list[Agent], survivability: float = 0.8
) -> tuple[Agent, list[tuple[Agent, Agent]]]:
    """Pairs the most fit (smallest errors) agents. Removes least desireable from the pool.

    Args:
        agents (list[Agent]): List of current agents.
        survivability (float, optional): How large of aportion of agents will survive?. Defaults to 0.8.

    Returns:
        tuple[Agent,list[tuple[Agent,Agent]]]: Current best agent, list of paired agents.
    """
    agents = sorted(agents, key=operator.attrgetter("error"))
    current_best_agent = agents[0]
    paired_agents = list(pairwise(agents))[
        : int(np.floor((len(agents) / 2) * survivability))
    ]
    logger.info(
        "Paired %d agents and culled %d agents",
        len(paired_agents) * 2,
        2 * (len(agents) / 2 - len(paired_agents)),
    )
    logger.info(
        "Current best agent has coefficients y = %.2fx + %.2f and an error of %.2f",
        current_best_agent.coef[1],
        current_best_agent.coef[0],
        current_best_agent.error,
    )
    return current_best_agent, paired_agents


# Create offspring with balanced weighted coefficients of parents +  small mutations
def create_offspring(
    p1: Agent,
    p2: Agent,
    mutation_coefficient: Union[float, int] = 1,
    fertility_rate: Union[float, int] = 3,
) -> list[Agent]:
    """Creates children for two agents. Takes into account the mutation coefficient and fertility rate.

    Args:
        p1 (Agent): Parent 1
        p2 (Agent): Parent 2
        mutation_coefficient (Union[float,int], optional): Determines how much a child's coefficients can mutate. Defaults to 1.
        fertility_rate (Union[float,int], optional): Determines how many children on average a pair should have. Defaults to 3.

    Returns:
        list[Agent]: List of children produced by the parents.
    """
    W = np.array([p1.error, p2.error])
    W /= W.sum()
    n_children = int(np.random.normal(loc=fertility_rate, scale=2, size=1)[0])

    children = [
        Agent(
            name=id_generator(),
            coef=(p1.coef * W[0] + p2.coef * W[1])
            + mutation_coefficient * (np.random.normal(loc=1, scale=2, size=2) - 1),
        )
        for _ in range(n_children)
    ]
    return children


def create_generation(
    paired_agents: list(tuple[Agent, Agent]),
    mutation_coefficient: Union[float, int],
    fertility_rate: Union[float, int],
) -> list[Agent]:
    logger.info("Current generation has %d agents", len(paired_agents) * 2)
    logger.info("Creating the new generation...")
    offspring = [
        create_offspring(*pair, mutation_coefficient, fertility_rate)
        for pair in paired_agents
    ]
    offspring = [item for sublist in offspring for item in sublist]
    logger.debug(f"{offspring = }")
    logger.info("The new generation has %d agents", len(offspring))
    return offspring


# Repeat until x iterations
def evolution(
    N_initial_population: int = 100,
    N_iterations: int = 10,
    mutation_coefficient: Union[float, int] = 1,
    fertility_rate: Union[float, int] = 3,
) -> tuple[
    Iterable[Union[float, int]],
    Iterable[Union[float, int]],
    np.ndarray,
]:
    """Runs the evolutionary process.

    Args:
        N_initial_population (int, optional). Defaults to 100.
        N_iterations (int, optional). Defaults to 10.
        mutation_coefficient (Union[float,int], optional): Determines how much a child's coefficients can mutate. Defaults to 1.
        fertility_rate (Union[float,int], optional): Determines how many children on average a pair should have. Defaults to 3.

    Returns:
        tuple[Iterable[Union[float, int]],Iterable[Union[float, int]],np.ndarray]: x,y,best coefficients for each iteration.
    """
    # Initialise world
    x, y = create_data()
    agents = initialise_agents(n_coefs=2, n=N_initial_population)
    best_coefs = []
    # Iterate evolutions
    i = 0
    previous_error = 999999999
    while i < N_iterations:
        logger.info("Iteration number %d", i)
        compute_all_errors(x, y, agents)
        current_best_agent, paired_agents = pair_and_cull(
            agents=agents, survivability=0.8
        )
        delta = abs((current_best_agent.error - previous_error) / previous_error)
        if delta < 0.05:
            logger.info("Stopping iterations, threshold error %.2f reached", delta)
            break
        best_coefs.append(current_best_agent.coef)
        del agents
        agents = create_generation(paired_agents, mutation_coefficient, fertility_rate)
        del paired_agents
        if len(agents) < 2:
            logger.info("Agents died out at iteration %d", i)
            break
        i += 1
    return x, y, np.array(best_coefs)


logger = init_logger(level_=logging.INFO, log_file="genetic_LR.log")
seed_all(421)

# In between, plot animation with Dash agents reaching the optimal solution
