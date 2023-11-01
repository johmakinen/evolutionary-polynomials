"""Visuals"""
import plotly.graph_objects as go
from typing import Iterable,Union
import numpy as np

def visualise_evolution(x:Iterable[Union[float, int]],y:Iterable[Union[float, int]],best_coefs:np.ndarray):
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