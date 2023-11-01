# pylint: skip-file
# Imports
import numpy as np
import random
import string
from dataclasses import dataclass, field
from itertools import pairwise
import logging
import os
import operator

# Logging
import logging

import plotly.graph_objects as go



def init_logger(level_=logging.INFO, log_file="genetic_LR.log"):
    os.remove(log_file)

    logger = logging.getLogger(__name__)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)
    logger.setLevel(level_)  # <<< Added Line
    c_handler.setLevel(level_)
    f_handler.setLevel(level_)

    # Create formatters and add it to handlers
    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Seeded everything with seed %s" % seed)


@dataclass(order=True)
class Agent:
    name: str
    coef: list  # [b,x0,x1,x2,...]
    error: float = np.inf
    sort_index: int = field(init=False)

    def __post_init__(self):
        self.sort_index = self.error


# Create data
def create_data(N: int = 100) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(0, N)
    k = 3.14
    b = 7
    y = k * x + b
    y += np.random.normal(scale=50, size=y.shape)
    logger.info(
        "Created %d datapoints with coefficients with y = %.2fx + %.2f" % (N, k, b)
    )
    return x, y


def id_generator(size=20, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


# Create 10 agents
def initialise_agents(n_coefs, n: int = 10):
    agents = [
        Agent(name=id_generator(), coef=np.random.randint(-10, 10, size=n_coefs))
        for i in range(n)
    ]
    logger.info("Created %d initial agents" % n)
    return agents


# Compute error of an agent
def compute_error(x, y, agent: Agent):
    agent.error = np.sum((y - (agent.coef[0] + agent.coef[1] * x)) ** 2)
    agent.__post_init__()


def compute_all_errors(x, y, agents: list[Agent]):
    for agent in agents:
        compute_error(x, y, agent)


# Pair agents, Kill 10% worst pairs to limit resources
def pair_and_cull(agents: list[Agent], survivability=0.8):
    agents = sorted(agents, key=operator.attrgetter("error"))
    current_best_agent = agents[0]
    paired_agents = list(pairwise(agents))[
        : int(np.floor((len(agents) / 2) * survivability))
    ]
    logger.info(
        "Paired %d agents and culled %d agents"
        % (len(paired_agents) * 2, 2 * (len(agents) / 2 - len(paired_agents)))
    )
    logger.info(
        "Current best agent has coefficients y = %.2fx + %.2f and an error of %.2f"
        % (
            current_best_agent.coef[1],
            current_best_agent.coef[0],
            current_best_agent.error,
        )
    )
    return current_best_agent, paired_agents


# Create offspring with balanced weighted coefficients of parents +  small mutations
def create_offspring(p1: Agent, p2: Agent, mutation_coefficient=1, fertility_rate=3):
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
    paired_agents: list(tuple[Agent, Agent]), mutation_coefficient, fertility_rate
):
    logger.info("Current generation has %d agents" % (len(paired_agents) * 2))
    logger.info("Creating the new generation...")
    offspring = [
        create_offspring(*pair, mutation_coefficient, fertility_rate)
        for pair in paired_agents
    ]
    offspring = [item for sublist in offspring for item in sublist]
    logger.debug(f"{offspring = }")
    logger.info("The new generation has %d agents" % len(offspring))
    return offspring


# Repeat until x iterations
def evolution(
    N_initial_population: int = 100,
    N_iterations: int = 10,
    mutation_coefficient=1,
    fertility_rate=3,
):
    # Initialise world
    x, y = create_data()
    agents = initialise_agents(n_coefs=2, n=N_initial_population)
    best_coefs = []
    # Iterate evolutions
    i = 0
    previous_error = 999999999
    while (i < N_iterations):
        logger.info('Iteration number %d' % i)
        compute_all_errors(x, y, agents)
        current_best_agent, paired_agents = pair_and_cull(
            agents=agents, survivability=0.8
        )
        delta = abs((current_best_agent.error - previous_error)/previous_error)
        if delta < 0.05:
            logger.info("Stopping iterations, threshold error %.2f reached" % delta)
            break
        best_coefs.append(current_best_agent.coef)
        del agents
        agents = create_generation(paired_agents, mutation_coefficient, fertility_rate)
        del paired_agents
        if len(agents) < 2:
            logger.info("Agents died out at iteration %d" % i)
            break
        i += 1
    return x,y, np.array(best_coefs)

def visualise_evolution(x,y,best_coefs):
        y_lines = x*(best_coefs[:,1].reshape(-1,1))+best_coefs[:,0].reshape(-1,1)


        # create the scatter plot
        points = go.Scatter(x=x, y=y, mode='markers')

        # create initial line
        line = go.Scatter(x=x, y=y_lines[0])

        # create a layout with out title (optional)
        layout = go.Layout(title_text="Gradient Descent Animation")

        # combine the graph_objects into a figure
        fig = go.Figure(data=[points, line])

        # create a list of frames
        frames = []

        # create a frame for every line y
        for i,y_line in enumerate(y_lines):

            # update the line
            line = go.Scatter(x=x, y=y_line)
            
            # create the button
            button = {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 150}}],
                    }
                ],
            }

            # add the button to the layout and update the
            # title to show the gradient descent step
            layout = go.Layout(updatemenus=[button],
                            title_text=f"Gradient Descent Step {i}")

            # create a frame object
            frame = go.Frame(
                data=[points, line],
                layout=go.Layout(title_text=f"Gradient Descent Step {i}")
            )

            # add the frame object to the frames list
            frames.append(frame)

        # combine the graph_objects into a figure
        fig = go.Figure(data=[points, line], frames=frames, layout=layout);

        # show our animation!
        fig.show()

# In between, plot animation with Dash agents reaching the optimal solution

if __name__ == "__main__":
    logger = init_logger(level_=logging.INFO, log_file="genetic_LR.log")
    seed_all(421)
    config = {
        "N_initial_population": 100,
        "N_iterations": 100,
        "mutation_coefficient": 0.1,
        "fertility_rate": 3,
    }
    x,y, best_coefs = evolution(**config)

    visualise_evolution(x,y, best_coefs)


# TODO:
# - Initialise git repo
# - Add subfolders and modularise code
# - Add possibility to input any polynomial
# - Add possibility for user to input polynomial in dash
# - Deploy
# - Add into CV/github.io