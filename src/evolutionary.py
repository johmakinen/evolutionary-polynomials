"Main evolutionary tools"
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
    if os.path.exists(log_file):
        os.remove(log_file)

    with open(log_file, mode="w", encoding="utf-8"):
        pass

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

    def predict_(self, x, use_bias):
        if not use_bias:
            y_pred = np.array(self.coef) * np.power(
                np.array([x] * len(self.coef)).T, np.arange(1, len(self.coef) + 1, 1)
            )
        else:
            y_pred = np.array(self.coef)[0] + np.power(
                np.array([x] * len(self.coef[1:])).T,
                np.arange(1, len(self.coef[1:]) + 1, 1),
            )
        y_pred = np.sum(y_pred, axis=1)
        return y_pred


# Create data
def create_data(
    bias: bool = False, degree: int = 1, N: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Creates the data from a given polynomial.

    Args:
        N (int, optional): How many data points are wanted. Defaults to 100.
        bias (bool, optional): Use bias term?. Defaults to 100.
        degree (int, optional): How many degrees in the polynomial?. Defaults to 1.

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    x = np.linspace(-10,10,N)
    k = np.random.normal(scale=2, size=degree)
    b = 15 if bias else 0
    perfect_agent = Agent(name="perfektio", coef=[b, *k])

    y = perfect_agent.predict_(x, use_bias=bias)
    del perfect_agent
    # Add noise
    y += np.random.normal(loc=y.mean(), scale=y.std(), size=y.shape)
    logger.info("Created %d datapoints with coefficients with y = %sx + %s", N, k, b)

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
        Agent(name=id_generator(), coef=np.random.normal(loc=0,scale=10, size=n_coefs))
        for _ in range(n)
    ]
    logger.info("Created %d initial agents", n)
    return agents


# Compute error of an agent
def compute_error(
    x: Iterable[Union[float, int]],
    y: Iterable[Union[float, int]],
    agent: Agent,
    use_bias: bool,
) -> None:
    """Computed the error for a single agent. Currently uses simple ordinary least squares

    Args:
        x (Iterable[Union[float,int]]): x values
        y (Iterable[Union[float,int]]): y values
        agent (Agent)
        use_bias (bool): Is bias term used?
    """
    agent.error = (1 / len(x)) * np.sum((y - agent.predict_(x, use_bias)) ** 2)
    agent.__post_init__()


def compute_all_errors(x, y, agents: list[Agent], use_bias: bool):
    for agent in agents:
        compute_error(x, y, agent, use_bias)


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
        "Current best agent has coefficients %s and an error of %.2f",
        current_best_agent.coef,
        current_best_agent.error,
    )
    return current_best_agent, paired_agents


def compute_child_coefs(
    p1: Agent, p2: Agent, mutation_coefficient: Union[float, int] = 1
) -> list[float]:
    W = np.array([p1.error, p2.error])
    W /= W.sum()

    res_coefs = (p1.coef * W[0] + p2.coef * W[1]) + mutation_coefficient * (
        np.random.normal(scale=1, size=len(p1.coef))
    )
    return res_coefs


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

    n_children = int(np.random.normal(loc=fertility_rate, scale=2, size=1)[0])

    children = [
        Agent(
            name=id_generator(), coef=compute_child_coefs(p1, p2, mutation_coefficient)
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
    logger.info("The new generation has %d agents", len(offspring))
    return offspring


# Repeat until x iterations
def evolution(
    polynomial: str,
    use_bias: bool,
    n_degree: int,
    n_coefs: int,
    N_initial_population: int = 100,
    N_iterations: int = 10,
    mutation_coefficient: Union[float, int] = 1,
    fertility_rate: Union[float, int] = 3,
) -> tuple[Iterable[Union[float, int]], Iterable[Union[float, int]], np.ndarray]:
    """Runs the evolutionary process.

    Args:
        polynomial (int): String representation of the wanted polynomial.
        use_bias (bool): Use a bias term?
        n_degree (int): What degree polynomial to fit?
        n_coefs (int): How many coefficients are used. 2 for y ~x+b,
        N_initial_population (int, optional). Defaults to 100.
        N_iterations (int, optional). Defaults to 10.
        mutation_coefficient (Union[float,int], optional): Determines how much a child's coefficients can mutate. Defaults to 1.
        fertility_rate (Union[float,int], optional): Determines how many children on average a pair should have. Defaults to 3.

    Returns:
        tuple[Iterable[Union[float, int]],Iterable[Union[float, int]],np.ndarray]: x,y,best coefficients for each iteration.
    """
    # Initialise world
    x, y = create_data(bias=use_bias, degree=n_degree)
    agents = initialise_agents(n_coefs=n_coefs, n=N_initial_population)
    best_coefs = []
    # Iterate evolutions
    i = 0
    previous_error = 999999999
    while i < N_iterations:
        logger.info("Iteration number %d", i)
        compute_all_errors(x, y, agents, use_bias)
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


logger = init_logger(level_=logging.DEBUG, log_file="genetic_logs.log")
seed_all(32)
