"""Visuals"""
from typing import Iterable
from typing import Union

import numpy as np
import plotly.graph_objects as go

from src.evolutionary import Agent

# logger = init_logger(level_=logging.WARNING, log_file="genetic_logs.log")
# seed_all(421)


def visualise_evolution(
    x: Iterable[Union[float, int]],
    y: Iterable[Union[float, int]],
    best_coefs: np.ndarray,
    use_bias: bool,
    ret=True,
):
    """Function to visualise the results of the evolutions

    Args:
        x (Iterable[Union[float, int]]): x data
        y (Iterable[Union[float, int]]): y data
        best_coefs (np.ndarray): array of best coefficients for each iteration
        use_bias (bool): Whether there is a bias term in the coefficient matrix
    """

    # Create the predicted lines using a perfect agent
    def find_nearest1(array, value):
        idx, _ = min(enumerate(array), key=lambda x: abs(x[1] - value))
        return idx

    y_lines = np.zeros(shape=(len(x), len(best_coefs[:, 0])))
    fill_ = len(x) < 20
    if fill_:
        # With small numer of points, the line will be discontinuous. Fill it with more values
        min_fill_points = 100
        y_filled_lines = np.zeros(shape=(min_fill_points, len(best_coefs[:, 0])))
        filled_x = np.linspace(start=min(x), stop=max(x), num=min_fill_points)
        idx_filled = (
            []
        )  # We need to keep track what points are given, and what are filled (for plotting errorbars)
        for x_ in x:
            idx_ = find_nearest1(filled_x, x_)
            idx_filled.append(idx_)
            filled_x[idx_] = x_

    best_agent = Agent(name="best", coef=[0] * len(best_coefs))
    for i, row in enumerate(best_coefs):
        best_agent.coef = row
        y_lines[:, i] = best_agent.predict_(x, use_bias)
        if fill_:
            y_filled_lines[:, i] = best_agent.predict_(filled_x, use_bias)

    y_lines = y_lines.T
    if fill_:
        y_filled_lines = y_filled_lines.T
    # create the scatter plot
    points = go.Scatter(x=x, y=y, mode="markers", name="Data")

    # create initial line
    line = go.Scatter(x=x, y=y_lines[0], name=f"Iteration {0}")

    # create a layout with out title (optional)
    layout = go.Layout(
        title_text="Evolutionary algorithm animation",
        yaxis=dict(range=[min(y) * 1.1, max(y) * 1.1]),
        xaxis=dict(range=[min(x) * 1.1, max(x) * 1.1]),
    )

    # combine the graph_objects into a figure
    fig = go.Figure(data=[points, line])

    # create a list of frames
    frames = []

    # create a frame for every line y
    for i, y_line in enumerate(y_lines):
        neg_error = np.clip(np.array(y - y_line), 0, np.inf)
        pos_error = np.abs(np.clip(np.array(y - y_line), -np.inf, 0))

        # update the line
        if fill_:
            filled_neg_error = np.zeros(len(filled_x))
            filled_neg_error[idx_filled] = neg_error
            filled_pos_error = np.zeros(len(filled_x))
            filled_pos_error[idx_filled] = pos_error

            line = go.Scatter(
                x=filled_x,
                y=y_filled_lines[i],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": filled_neg_error,
                    "arrayminus": filled_pos_error,
                    "color": "#ff0000",
                    "width": 0.1,
                },
                line={"color": "#b900ff"},
                name=f"Iteration {i}",
            )
        else:
            line = go.Scatter(
                x=x,
                y=y_line,
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": neg_error,
                    "arrayminus": pos_error,
                    "color": "#ff0000",
                },
                line={"color": "#b900ff"},
                name=f"Iteration {i}",
            )

        # create the button
        button = {
            "type": "buttons",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 500}}],
                }
            ],
        }

        # add the button to the layout and update the
        # title to show the step
        layout = go.Layout(updatemenus=[button], title_text=f"Evolution Step {i}")
        # create a frame object
        frame = go.Frame(
            data=[points, line],
            layout=go.Layout(title_text=f"Evolution Step {i}"),
        )

        # add the frame object to the frames list
        frames.append(frame)

    # combine the graph_objects into a figure
    fig = go.Figure(data=[points, line], frames=frames, layout=layout)

    if ret:
        return fig

    else:  # show our animation!
        fig.show()
