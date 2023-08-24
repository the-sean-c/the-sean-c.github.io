# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#  | include: false
import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from utilities.plot_template import the_template

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| fig-cap: "Life Expectancy and GDP"

# Parameters
mean = 0.95
lambda_param = 1.0 / mean
x = np.linspace(0, 6, 400)  # Generate x values
y = lambda_param * np.exp(-lambda_param * x)  # Exponential distribution function

# Sample 50 points and jitter for dodging
sample_points_x = np.random.exponential(mean, 50)
jitter = 0.05  # Adjust this value for more/less jitter
sample_points_y = [np.random.uniform(-jitter, jitter) for _ in range(50)]

# Plot using Plotly
fig = go.Figure()

# Add the curve
fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="exponential_distribution"))

# Add the sample points along the x-axis with jitter
fig.add_trace(
    go.Scatter(
        x=sample_points_x, y=sample_points_y, mode="markers", name="sample_points"
    )
)

# Add line at the mean
fig.add_shape(
    go.layout.Shape(
        type="line",
        x0=mean,
        x1=mean,
        y0=0,
        y1=lambda_param * np.exp(-lambda_param * mean),
        line=dict(color="Red", width=2),
    )
)

# Add arrow annotation pointing to the mean
fig.add_annotation(
    x=mean,
    y=lambda_param
    * np.exp(-lambda_param * mean)
    / 2,  # Place the arrow halfway along the mean line
    text="expected mean",
    showarrow=True,
    arrowhead=4,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="Red",
    ax=80,
    ay=-80,
)

# Layout
fig.update_layout(
    xaxis_title="Reward ($)",
    yaxis_title="Probability Density",
    yaxis=dict(showgrid=False, ticklen=5, tickwidth=1),
    xaxis=dict(showgrid=False, ticklen=5, tickwidth=1),
    template=the_template,
)

fig.show()
#
#
#
#
#
#
n_machines = 3
n_periods = 500

sigma = 0.1
cost_per_game = 1
expected_prize = 0.95
starting_value = 100

mean_lin = np.ones(n_machines) * expected_prize
var_lin = np.ones(n_machines) * (sigma**2)

mean_log = np.log(mean_lin**2 / (np.sqrt(mean_lin**2 + var_lin)))
var_log = np.log(1 + var_lin / mean_lin**2)
cov_log = np.diag(var_log)
#
#
#
#
rng = np.random.default_rng(2)
e_reward_c = np.exp(rng.multivariate_normal(mean_log, cov_log, size=(1))[0])
e_reward_c = np.tile(e_reward_c, (n_periods, 1))
reward_c = np.random.exponential(e_reward_c)

machine_labels = [str(i + 1) for i in range(reward_c.shape[1])]

rewards_df = (
    pl.DataFrame({"turn": pl.Series(values=range(1, reward_c.shape[0] + 1))})
    .with_columns(pl.from_numpy(reward_c, schema=machine_labels, orient="row"))
    .melt(
        id_vars="turn",
        value_vars=machine_labels,
        value_name="reward",
        variable_name="machine",
    )
).with_columns(
    (
        pl.DataFrame(
            {"turn": pl.Series(values=range(1, reward_c.shape[0] + 1))}
        ).with_columns(pl.from_numpy(e_reward_c, schema=machine_labels, orient="row"))
    ).melt(
        id_vars="turn",
        value_vars=machine_labels,
        value_name="expected_reward",
        variable_name="machine",
    )
)

fig = px.scatter(
    rewards_df, x="turn", y="reward", color="machine", template=the_template
)

fig.show()

#
#
#
