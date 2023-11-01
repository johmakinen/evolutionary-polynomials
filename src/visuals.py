"""Visuals"""
import sys
from typing import Iterable, Union
import numpy as np
import logging
from src.evolutionary import init_logger, Agent

import plotly.graph_objects as go

# logger = init_logger(level_=logging.WARNING, log_file="genetic_logs.log")
# seed_all(421)


def visualise_evolution(
    x: Iterable[Union[float, int]],
    y: Iterable[Union[float, int]],
    best_coefs: np.ndarray,
    use_bias: bool,
):
    # Create the predicted lines using a perfect agent

    y_lines = np.zeros(shape=(len(x), len(best_coefs[:, 0])))

    best_agent = Agent(name="best", coef=[0] * len(best_coefs))
    for i, row in enumerate(best_coefs):
        best_agent.coef = row
        y_lines[:, i] = best_agent.predict_(x, use_bias)
    y_lines = y_lines.T

    # create the scatter plot
    points = go.Scatter(x=x, y=y, mode="markers")

    # create initial line
    line = go.Scatter(x=x, y=y_lines[0])

    # create a layout with out title (optional)
    layout = go.Layout(title_text="Evolutionary algorithm animation")

    # combine the graph_objects into a figure
    fig = go.Figure(data=[points, line])

    # create a list of frames
    frames = []

    # create a frame for every line y
    for i, y_line in enumerate(y_lines):
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
        layout = go.Layout(updatemenus=[button], title_text=f"Evolution Step {i}")

        # create a frame object
        frame = go.Frame(
            data=[points, line], layout=go.Layout(title_text=f"Evolution Step {i}")
        )

        # add the frame object to the frames list
        frames.append(frame)

    # combine the graph_objects into a figure
    fig = go.Figure(data=[points, line], frames=frames, layout=layout)

    # show our animation!
    fig.show()
