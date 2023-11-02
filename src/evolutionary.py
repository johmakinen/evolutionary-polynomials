"Main evolutionary tools"
# Imports
import random
import string
from dataclasses import dataclass, field
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
    coef: Iterable[Union[float, int]]  # [b,x0,x1,x2,...]
    error: float = np.inf
    sort_index: int = field(init=False)

    def __post_init__(self):
        self.sort_index = self.error

    def predict_(self, x, use_bias):
        if not use_bias:
            y_pred = np.array(self.coef) * np.power(
                np.array([x] * len(self.coef)).T, np.arange(1, len(self.coef) + 1, 1)
            )
            y_pred = np.sum(y_pred, axis=1)
        else:
            y_pred = np.array(self.coef)[0] + np.sum(
                np.array(self.coef[1:])
                * np.power(
                    np.array([x] * len(self.coef[1:])).T,
                    np.arange(1, len(self.coef[1:]) + 1, 1),
                ),
                axis=1,
            )

        return y_pred


@dataclass
class Population:
    """A Dataclass for a population of agents.
    Each Population has:
    - Name
    - List of agents within the population
    - Population size
    - Population level fertility rate
    - Population level mutation coefficient
    - Population level survivability
    """

    name: str
    agents: list[Agent]
    size: int = 0
    fertility_rate: Union[float, int] = 3
    mutation_coefficient: Union[float, int] = 0.4
    survivability: Union[float, int] = 0.8


class World:
    """Class for the world. Within the world there are population(s). Each world tries to solve one polynomial. You can evolve the world in time, and consequently evolve the population(s).
    Only one population per world for now.
    """

    def __init__(
        self,
        name: str,
        polynomial: str,
        use_bias: bool = True,
        mutation_coefficient: Union[float, int] = 1,
        fertility_rate: Union[float, int] = 3,
        survivability: Union[float, int] = 0.8,
        N_max_iter: int = 50,
        N_initial_population: int = 100,
        N_iterations: int = 10,
        N_datapoints: int = 50,
    ):
        self.name = name
        self.polynomial = polynomial
        self.use_bias = use_bias
        self.mutation_coefficient = mutation_coefficient
        self.fertility_rate = fertility_rate
        self.survivability = survivability
        self.N_max_iter = N_max_iter
        self.N_initial_population = N_initial_population
        self.N_iterations = N_iterations
        self.N_datapoints = N_datapoints

        self.population = []
        self.x = []
        self.y = []
        self.n_degree = 0
        self.n_coefs = []
        self.best_coefs = []
        self.current_iterarion: int = 0

    def initialise_world_(
        self,
        x: Iterable[Union[float, int]] = None,
        y: Iterable[Union[float, int]] = None,
    ):
        self.n_degree = self.polynomial.count("x")
        self.n_coefs = self.n_degree + 1 if self.use_bias else self.n_degree
        if (not x) or (not y):
            self.x, self.y = create_data(
                bias=self.use_bias, degree=self.n_degree, N=self.N_datapoints
            )
            # TODO: N_datapoints is not optional here
        else:
            self.x = x
            self.y = y
            # TODO: N_datapoints is optional here (not used)

        agents = initialise_agents(n_coefs=self.n_coefs, n=self.N_initial_population)
        self.population = Population(
            "pop1",
            agents=agents,
            size=len(agents),
            fertility_rate=self.fertility_rate,
            mutation_coefficient=self.mutation_coefficient,
            survivability=self.survivability,
        )

    def evolve_(self):
        # Iterate evolutions
        previous_error = 999999999
        while self.current_iterarion < self.N_max_iter:
            logger.info("Iteration number %d", self.current_iterarion)
            compute_all_errors(self.x, self.y, self.population.agents, self.use_bias)
            current_best_agent, paired_agents = pair_and_cull(
                self.population.agents, survivability=self.survivability
            )

            delta = abs((current_best_agent.error - previous_error) / previous_error)
            if delta < 0.05:
                logger.info("Stopping iterations, threshold error %.2f reached", delta)
                break

            self.best_coefs.append(current_best_agent.coef)

            agents = create_generation(
                paired_agents, self.mutation_coefficient, self.fertility_rate
            )
            self.population.agents = agents

            if len(agents) < 2:
                logger.info("Agents died out at iteration %d", self.current_iterarion)
                break

            self.current_iterarion += 1
        return self.x, self.y, np.array(self.best_coefs)


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
    x = np.linspace(-10, 10, N)
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
        Agent(name=id_generator(), coef=np.random.normal(loc=0, scale=10, size=n_coefs))
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
    return agent.error


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
    paired_agents = [
        (agents[i], agents[i + 1]) for i in range(0, len(agents), 2)
    ] 

    paired_agents = paired_agents[
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
    W = 1 - W / W.sum()  # Inverse weights, less error = more weight

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


logger = init_logger(level_=logging.INFO, log_file="genetic_logs.log")
seed_all(32)
