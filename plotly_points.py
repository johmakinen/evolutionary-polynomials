import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash
from dash import dcc
from dash import html
from dash import Input
from dash import Output
from dash import State
from dash.exceptions import PreventUpdate

from src.evolutionary import World
from src.visuals import visualise_evolution

# create image and plotly express object
init_xrange = [-50, 50]
init_yrange = [-25, 25]
# fig = px.imshow(np.zeros(shape=(init_yrange[1],init_xrange[1], 4)), origin="lower")
# fig.add_scatter(x=[0], y=[0], mode="markers", marker_color="purple", marker_size=5)
fig = go.Figure(
    go.Image(
        z=np.zeros(shape=(init_yrange[1] * 3, init_xrange[1] * 3, 4)),
        opacity=0,
        dx=1,
        dy=1,
        x0=-init_xrange[1] * 3 / 2,
        y0=-init_yrange[1] * 3 / 2,
    )
)
fig.add_trace(
    go.Scatter(
        x=[0],
        y=[0],
        mode="markers",
        marker_color="purple",
        marker_size=5,
        name="init_point",
    )
)
fig.add_trace(
    go.Scatter(x=[0], y=[0], mode="markers", marker_color="white", marker_size=5)
)

# fig.add_scatter(x=[0], y=[0], mode="markers", marker_color="purple", marker_size=0.1 if first_ else 5)
first_ = False
fig.update_yaxes(range=init_yrange)
xs = []
ys = []

# update layout
layout = go.Layout(title_text="Evolutionary algorithm animation", width=20)
fig.update_layout(layout)
fig.update_traces(hovertemplate=None, hoverinfo="none")
# hide color bar
fig.update_coloraxes(showscale=False)

# Build App
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)

# app layout
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    dcc.Graph(
                        id="graph",
                        figure=fig,
                        config={
                            "scrollZoom": True,
                            "displayModeBar": False,
                        },
                    )
                ],
                width={"size": 10, "offset": 0},
            ),
            justify="around",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.A(
                            html.Button("Refresh Page", id="refresh_button"), href="/"
                        ),
                        html.A(html.Button("Evolve", id="evolve_button", n_clicks=0)),
                        dcc.Markdown(
                            """
                            # Functionality:
                            - click anywhere on the image, shape is created at click position
                            - use the "Refresh Page" button to reload the image
                            """
                        ),
                    ],
                    width={"size": 5, "offset": 0},
                ),
            ],
            justify="around",
        ),
    ],
    fluid=True,
)


# xs.extend([x])
# ys.extend([y])
@app.callback(
    Output("graph", "figure"), State("graph", "figure"), Input("graph", "clickData")
)
def get_click(graph_figure, clickData):
    if not clickData:
        raise PreventUpdate
    else:
        points = clickData.get("points")[0]
        x = points.get("x")
        y = points.get("y")
        xs.extend([x])
        ys.extend([y])
        # get scatter trace (in this case it's the last trace)
        scatter_x, scatter_y = [
            graph_figure["data"][1].get(coords) for coords in ["x", "y"]
        ]
        scatter_x.append(x)
        scatter_y.append(y)

        print(graph_figure["data"][1])
        # update figure data (in this case it's the last trace)
        graph_figure["data"][1].update(x=scatter_x)
        graph_figure["data"][1].update(y=scatter_y)
        # graph_figure["data"][1].update(marker={'color': 'purple', 'size': 5})
    return graph_figure


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    State("graph", "figure"),
    Input("evolve_button", "n_clicks"),
    prevent_initial_call=True,
)
def visualise_evolution_(graph_figure, n_clicks):
    if not n_clicks:
        raise PreventUpdate

    polynomial = "y ~ x+x^2+x^3+b"  # only 2d for now
    use_bias = True

    config = {
        "N_initial_population": 100,
        "N_max_iter": 50,
        "mutation_coefficient": 0.5,
        "fertility_rate": 3,
    }
    config["polynomial"] = polynomial
    config["use_bias"] = use_bias

    # Backend
    world = World(name="pop1", **config)
    world.initialise_world_(x=xs, y=ys)
    x, y, best_coefs = world.evolve_()
    graph_figure = visualise_evolution(x, y, best_coefs, use_bias)
    xs.clear()
    ys.clear()
    # graph_figure["layout"]["yaxis"].update(autorange=True)

    return graph_figure


def create_shape(x, y, size=4, color="rgba(39,43,48,255)"):
    """
    function creates a shape for a dcc.Graph object

    Args:
        x: x coordinate of center point for the shape
        y: y coordinate of center point for the shape
        size: size of annotation (diameter)
        color: (rgba / rgb / hex) string or any other color string recognized by plotly

    Returns:
        a list containing a dictionary, keys corresponding to dcc.Graph layout update
    """
    shape = [
        {
            "editable": True,
            "xref": "x",
            "yref": "y",
            "layer": "above",
            "opacity": 1,
            "line": {"color": color, "width": 1, "dash": "solid"},
            "fillcolor": color,
            "fillrule": "evenodd",
            "type": "circle",
            "x0": x - size / 2,
            "y0": y - size / 2,
            "x1": x + size / 2,
            "y1": y + size / 2,
        }
    ]
    return shape


if __name__ == "__main__":
    app.run_server(debug=True, port=8053)
