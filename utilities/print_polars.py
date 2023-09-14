import plotly.graph_objects as go

from utilities.plot_template import the_template


def print_polars(df):
    fig = go.Figure(
        data=[
            go.Table(header={"values": df.columns}, cells={"values": df.to_numpy().T})
        ],
    )
    fig.update_layout(margin=dict(r=5, l=5, t=5, b=5))

    fig.show()
