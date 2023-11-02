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
# img = np.random.randint(0, 255, (90, 160))
# fig = px.imshow(np.zeros(shape=(100,160)), color_continuous_scale='gray',origin='lower')
# fig = px.scatter()
# create image and plotly express object
fig = px.imshow(np.zeros(shape=(90, 160, 4)), origin="lower", extent=(-10, 10, -5, 5))
fig.add_scatter(x=[0], y=[0], mode="markers", marker_color="purple", marker_size=5)
# fig = go.Figure(layout_yaxis_range=[-20,20],layout_xaxis_range=[-20,20])
xs = []
ys = []

# update layout
layout = go.Layout(title_text="Evolutionary algorithm animation")
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
                    ),
                    dcc.Graph(
                        id="graph2",
                        figure=fig,
                        config={
                            "scrollZoom": True,
                            "displayModeBar": False,
                        },
                    ),
                ],
                width={"size": 5, "offset": 0},
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

        # update figure data (in this case it's the last trace)
        graph_figure["data"][1].update(x=scatter_x)
        graph_figure["data"][1].update(y=scatter_y)

    return graph_figure


@app.callback(
    Output("graph2", "figure"),
    State("graph2", "figure"),
    Input("evolve_button", "n_clicks"),
)
def visualise_evolution_(graph_figure, n_clicks):
    if not n_clicks:
        raise PreventUpdate

    polynomial = "y ~ x+b"  # only 2d for now
    use_bias = True

    config = {
        "N_initial_population": 100,
        "N_max_iter": 50,
        "mutation_coefficient": 5,
        "fertility_rate": 3,
    }
    config["polynomial"] = polynomial
    config["use_bias"] = use_bias

    # Backend
    world = World(name="pop1", **config)
    world.initialise_world_(x=xs, y=ys)
    x, y, best_coefs = world.evolve_()
    graph_figure = visualise_evolution(x, y, best_coefs, use_bias)
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


# TODO: Evolve page is same but use visuals
# TODO: Modify both pages to look alike

if __name__ == "__main__":
    app.run_server(debug=True, port=8053)
