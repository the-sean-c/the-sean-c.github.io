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
#  | include: false
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
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
#| echo: false

n_machines = 3
n_periods = 800

sigma = 0.2
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
# | echo: false
# Mean-reverting process
rng = np.random.default_rng(14)
noise_log = rng.multivariate_normal(np.zeros(n_machines), cov_log, size=(n_periods + 200))

e_rewards_log = np.ones_like(noise_log) * mean_log
theta = 0.01
for i in range(1, n_periods + 200):
    e_rewards_log[i, :] = e_rewards_log[i - 1, :] + (
        0.01 * (mean_log - e_rewards_log[i - 1, :]) + 0.15 * noise_log[i, :]
        )
e_rewards = np.exp(e_rewards_log[200:])

# Generate rewards from expected rewards
rewards = rng.exponential(e_rewards)

machine_labels = [str(i + 1) for i in range(rewards.shape[1])]

rewards_df = pd.DataFrame(rewards)
rewards_df.columns = machine_labels
rewards_df["turn"] = list(range(1, rewards.shape[0] + 1))
rewards_df = rewards_df.melt(
    id_vars="turn",
    value_vars=machine_labels,
    value_name="reward",
    var_name="machine",)

e_rewards_df = pd.DataFrame(e_rewards)
e_rewards_df.columns = machine_labels
e_rewards_df["turn"] = list(range(1, e_rewards.shape[0] + 1))
e_rewards_df = e_rewards_df.melt(
    id_vars="turn",
    value_vars=machine_labels,
    value_name="expected_reward",
    var_name="machine",)

rewards_df = rewards_df.merge(e_rewards_df, on=("turn", "machine"), how="left")
#
#
#
#| echo: false
#| label: tbl-6_turns
#| tbl-cap: Results from 6 Turns
# Get the number of rows
num_rows = 6
# Randomly select an entry from each row
indices = [0, 1, 2, 0, 1, 2]
sample_rewards = rewards[np.arange(num_rows), indices]

df = pd.DataFrame(
    {
        "turn": list(range(1, num_rows + 1)),
        "machine_id": indices,
        "prize": sample_rewards,
    }
)

d = dict(selector="th",
    props=[('text-align', 'center')])

(df.style.hide()
    .format({'prize': '${:,.2f}'})
    .set_properties(**{'width':'10em', 'text-align':'center'})
    .set_table_styles([d]))

#
#
#
#
#
# | label: fig-exponential_distribution_pdf
# | echo: false
# | fig-cap: "Exponential Distribution"
# | fig-alt: "A plot of the exponential distribution probability density overlain by some sample points."

# Parameters
mean = 0.95
lambda_param = 1.0 / mean
x = np.linspace(0, 6, 400)  # Generate x values
y = lambda_param * np.exp(-lambda_param * x)  # Exponential distribution function

# Sample 50 points and jitter for dodging
sample_points_x = np.random.exponential(mean, 50)
jitter = 0.05  # Adjust this value for more/less jitter
sample_points_y = [np.random.uniform(-jitter, jitter) for _ in range(50)]

palette = sns.color_palette()

sns.lineplot(x=x, y=y)
plt.plot([mean, mean], [lambda_param * np.exp(-lambda_param * mean), 0], color=palette[0])
plt.annotate("expected mean", (mean, 0.2), (1.5, 0.5),
    arrowprops=dict(facecolor=palette[0], edgecolor=palette[0],
        arrowstyle="fancy,tail_width=0.07,head_width=0.7,head_length=1"))
plt.annotate("random sample", (1.5, 0), (4, 0.3),
    arrowprops=dict(facecolor=palette[0], edgecolor=palette[0],
        arrowstyle="fancy,tail_width=0.1,head_width=1,head_length=1"))
sns.scatterplot(x=sample_points_x, y=sample_points_y)
# plt.grid(axis='y', color='grey', linestyle='--', linewidth=0.5)
sns.despine(left=True, bottom=True, top=True, right=True)
plt.tick_params(axis='x', which='both', bottom=True, left=True)
plt.xlabel('prize')
plt.ylabel('probability density function')
plt.show()

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
# | label: fig-expected_reward
# | echo: false
# | fig-cap: "Expected Rewards for each of 3 Machines"
# | fig-alt: "A line-plot of the expected rewards to each of 3 machines."

fig = px.line(
    rewards_df, x="turn", y="expected_reward", color="machine", template=the_template
)
fig.update_layout(
    xaxis_title="turn",
    yaxis_title="expected reward ($)",
    yaxis=dict(showgrid=True, ticklen=5, tickwidth=1),
    xaxis=dict(showgrid=False, ticklen=5, tickwidth=1),
)
fig.show()
#
#
#
#
# | label: fig-reward
# | echo: false
# | fig-cap: "Actual Rewards for each of 3 Machines"
# | fig-alt: "A scatter plot of the rewards to each of 3 machines."

fig = px.scatter(
    rewards_df,
    x="turn",
    y="reward",
    color="machine",
    template=the_template,
)
# Layout
fig.update_layout(
    xaxis_title="turn",
    yaxis_title="reward ($)",
    yaxis=dict(showgrid=True, ticklen=5, tickwidth=1, ticks="outside"),
    xaxis=dict(
        showgrid=False, ticklen=5, tickwidth=1, zeroline=False, range=[0, n_periods]
    ),
)
fig.show()

#
#
#
#
#
#
#
