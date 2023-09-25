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
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
#| label: tbl-6_turns
#| tbl-cap: Results from 6 Turns

df = pd.DataFrame(
    {
        "turn": [1, 2, 3, 4, 5, 6],
        "machine_id": [1, 2, 3, 1, 2, 3],
        "prize": [0.10, 1.32, 0.29, 1.18, 1.10, 0.17]
    }
)

# Format table
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
# | label: fig-exponential_distribution_pdf
# | fig-cap: "Example of an Exponential Distribution"
# | fig-alt: "A plot of the exponential distribution probability density overlain by some sample points."

# Parameters
rng = np.random.default_rng(7)
mean = 0.99
lambda_param = 1.0 / mean
x = np.linspace(0, 6, 400)  # Generate x values
y = lambda_param * np.exp(-lambda_param * x)  # Exponential distribution function

# Sample 50 points and jitter for dodging
sample_points_x = rng.exponential(mean, 50)
jitter = 0.05  # Adjust this value for more/less jitter
sample_points_y = [rng.uniform(-jitter, jitter) for _ in range(50)]

palette = sns.color_palette()

sns.lineplot(x=x, y=y)
plt.plot([mean, mean], [lambda_param * np.exp(-lambda_param * mean), 0], color=palette[0])
plt.annotate("expected prize", (mean, 0.2), (1.5, 0.5),
    arrowprops=dict(facecolor=palette[0], edgecolor=palette[0],
        arrowstyle="simple,tail_width=0.07,head_width=0.7,head_length=1"))
plt.annotate("random samples", (2.5, 0), (4, 0.3),
    arrowprops=dict(facecolor=palette[0], edgecolor=palette[0],
        arrowstyle="simple,tail_width=0.07,head_width=0.7,head_length=1"))
sns.scatterplot(x=sample_points_x, y=sample_points_y)
# plt.grid(axis='y', color='grey', linestyle='--', linewidth=0.5)
sns.despine(left=True, bottom=True, top=True, right=True)
plt.tick_params(axis='x', which='both', bottom=True, left=True)

formatter = ticker.FuncFormatter(lambda x, pos: f'${x:.0f}')
plt.gca().xaxis.set_major_formatter(formatter)
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
# | label: fig-expected_reward
# | fig-cap: "Expected Prizes for each of 3 Machines"
# | fig-alt: "A line-plot of the expected prizes to each of 3 machines."

n_iterations = 5000
n_machines = 3
n_turns = 500

sigma = 0.2
cost_per_game = 1
expected_prize = 0.99
starting_value = 100

rng = np.random.default_rng(5)

mean_lin = np.ones(n_machines) * expected_prize
var_lin = np.ones(n_machines) * (sigma**2)

mean_log = np.log(mean_lin**2 / (np.sqrt(mean_lin**2 + var_lin)))
var_log = np.log(1 + var_lin / mean_lin**2) 
cov_log = np.diag(var_log)

# Mean-reverting process
noise_log = rng.multivariate_normal(np.zeros(n_machines), cov_log, size=(n_iterations, n_turns + 200))

e_rewards_log = np.ones_like(noise_log) * mean_log
theta = 0.01
for i in range(1, n_turns + 200):
    e_rewards_log[:, i, :] = e_rewards_log[:, i - 1, :] + (
        0.01 * (mean_log - e_rewards_log[:, i - 1, :]) + 0.15 * noise_log[:, i, :]
        )
e_rewards = np.exp(e_rewards_log[:, 200:])

# Generate rewards from expected rewards
rewards = rng.exponential(e_rewards)

machine_labels = [str(i + 1) for i in range(rewards.shape[2])]

rewards_df = pd.DataFrame(rewards[0, :, :])
rewards_df.columns = machine_labels
rewards_df["turn"] = list(range(1, rewards.shape[1] + 1))
rewards_df = rewards_df.melt(
    id_vars="turn",
    value_vars=machine_labels,
    value_name="prize",
    var_name="machine",)

e_rewards_df = pd.DataFrame(e_rewards[0, ::])
e_rewards_df.columns = machine_labels
e_rewards_df["turn"] = list(range(1, e_rewards.shape[1] + 1))
e_rewards_df = e_rewards_df.melt(
    id_vars="turn",
    value_vars=machine_labels,
    value_name="expected_prize",
    var_name="machine",)

rewards_df = rewards_df.merge(e_rewards_df, on=("turn", "machine"), how="left")

# Plot
ax = sns.lineplot(data=rewards_df, x="turn", y="expected_prize", hue="machine")
plt.grid(axis='y', color='#E5E5E5')
sns.despine(left=True, bottom=True, top=True, right=True)
plt.tick_params(axis='x', which='both', bottom=True, left=True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
plt.ylim(0.6, 1.8)

formatter = ticker.FuncFormatter(lambda x, pos: f'${x:.2f}')
plt.gca().yaxis.set_major_formatter(formatter)
plt.show()
#
#
#
#
#
# | label: fig-reward
# | fig-cap: "Actual Rewards for each of 3 Machines"
# | fig-alt: "A scatter plot of the rewards to each of 3 machines."

ax = sns.scatterplot(data=rewards_df, x="turn", y="prize", hue="machine", s=7)
plt.grid(axis='y', color='#E5E5E5')
sns.despine(left=True, bottom=True, top=True, right=True)
plt.tick_params(axis='x', which='both', bottom=True, left=True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
plt.ylim(0.0, 11)

formatter = ticker.FuncFormatter(lambda x, pos: f'${x:.2f}')
plt.gca().yaxis.set_major_formatter(formatter)
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
#
# | label: fig-expected_reward_oracle
# | fig-cap: "Expected Prizes under the Oracle Strategy"
# | fig-alt: "A scatter plot of the prizes won by the Oracle Strategy."

oracle_e_rewards = np.max(e_rewards[0], axis=1)
turns = list(range(1, len(oracle_e_rewards)+1))

ax = sns.lineplot(data=rewards_df, x="turn", y="expected_prize", hue="machine", alpha=0.3)
sns.lineplot(x=turns, y=oracle_e_rewards, color=palette[4], label="oracle")

plt.grid(axis='y', color='#E5E5E5')
sns.despine(left=True, bottom=True, top=True, right=True)
plt.tick_params(axis='x', which='both', bottom=True, left=True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.ylim(0.6, 1.8)

formatter = ticker.FuncFormatter(lambda x, pos: f'${x:.2f}')
plt.gca().yaxis.set_major_formatter(formatter)
plt.show()
#
#
#
#
#
#
#
# | label: fig-expected_reward_random
# | fig-cap: "Expected Prizes under the Random Strategy"
# | fig-alt: "A scatter plot of the prizes won by the Random Strategy."

random_choice = rng.integers(0, n_machines, (n_turns))
random_choice = np.expand_dims(random_choice, 1)
random_e_rewards = np.take_along_axis(e_rewards[0], random_choice, axis=1)
random_e_rewards = np.squeeze(random_e_rewards)

# random_e_rewards = rng.choice(e_rewards[0], axis=1)
turns = list(range(1, len(oracle_e_rewards)+1))

ax = sns.lineplot(data=rewards_df, x="turn", y="expected_prize", hue="machine", alpha=0.3)
sns.lineplot(x=turns, y=random_e_rewards, color=palette[5], label="random")

plt.grid(axis='y', color='#E5E5E5')
sns.despine(left=True, bottom=True, top=True, right=True)
plt.tick_params(axis='x', which='both', bottom=True, left=True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.ylim(0.6, 1.8)

formatter = ticker.FuncFormatter(lambda x, pos: f'${x:.2f}')
plt.gca().yaxis.set_major_formatter(formatter)
plt.show()
#
#
#
#
#
# | label: fig-average_reward_random
# | fig-cap: "Running Average of Prizes won by Oracle and Random Strategies over 5000 iterations"
# | fig-alt: "Line plot of the running Average of prizes won by Oracle and Random Strategies over 5000 iterations."

# Oracle Strategy
oracle_choice = np.argmax(e_rewards, axis=2, keepdims=True)
oracle_rewards = np.take_along_axis(rewards, oracle_choice, axis=2)
oracle_rewards = np.squeeze(oracle_rewards)

turns = np.array(list(range(1, n_turns+1)))
average_oracle_rewards = np.cumsum(oracle_rewards, axis=1) / turns
average_oracle_rewards = np.mean(average_oracle_rewards, axis=0)

# Random Strategy
random_choice = rng.integers(0, n_machines, (n_iterations, n_turns))
random_choice = np.expand_dims(random_choice, 2)
random_rewards = np.take_along_axis(rewards, random_choice, axis=2)
random_rewards = np.squeeze(random_rewards)

turns = np.array(list(range(1, random_rewards.shape[1]+1)))
average_random_rewards = np.cumsum(random_rewards, axis=1) / turns
average_random_rewards = np.mean(average_random_rewards, axis=0)

# Plotting
sns.lineplot(x=turns, y=average_oracle_rewards, color=palette[4], label="oracle")
sns.lineplot(x=turns, y=average_random_rewards, color=palette[5], label="random")

plt.grid(axis='y', color='#E5E5E5')
sns.despine(left=True, bottom=True, top=True, right=True)
plt.tick_params(axis='x', which='both', bottom=True, left=True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
plt.tight_layout(rect=[0, 0, 0.98, 1])
plt.ylim(0.6, 1.8)
plt.xlabel("turn")
plt.ylabel("average_prize")

formatter = ticker.FuncFormatter(lambda x, pos: f'${x:.2f}')
plt.gca().yaxis.set_major_formatter(formatter)
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
#
# | label: fig-average_reward_exploitation
# | fig-cap: "Running Average of Prizes won by the Exploitation Strategy over 5000 iterations"
# | fig-alt: "Line plot of the running Average of prizes won by the Exploitation Strategy over 5000 iterations."
def strategy(epsilon: float = 0.0, alpha: float=None, initial_value: float = None):
    """Runs a Strategy over a Bootstrap/Monte Carlo of rewards.

    Inputs:
    -------
    epsilon: float
        The probability that the user will try a different machine at random in a given game.
    alpha: float
        The step size parameter used for exponential weighted averaging. A value of None
        means that the mean is used.
    initial_value: float
        The prize expected by the player for each machine at the beginning of the game.
    """
    # Initialize vectors
    running_value_estimate = np.zeros((n_iterations, n_turns, n_machines))
    running_count = np.zeros((n_iterations, n_turns, n_machines), dtype=int)
    strategy_rewards = np.zeros((n_iterations, n_turns))
    running_selection = np.zeros((n_iterations, n_turns), dtype=int)
    selection = np.zeros(n_iterations, dtype=int)[:, None]

    # Instantiate all random variables up front
    random_selection = rng.integers(low=0, high=n_machines, size=(n_iterations, n_turns))
    random_explore = rng.uniform(0, 1, size=(n_iterations, n_turns))
    for i in range(n_turns):
        if i < n_machines:
            # Try all machines once.
            selection = np.array([i]*n_iterations)[:, None]
        else:
            # Explore with some probability epsilon
            explore = random_explore[:, i] < epsilon
            selection[explore] = random_selection[explore, i][:, None]
            # Otherwise, use greedy selection (select machine thought most valuable)
            selection[~explore] = np.argmax(running_value_estimate[~explore, i-1, :], axis=1)[:, None]

        running_selection[:, i] = selection[:, 0]

        strategy_rewards[:, i] = np.take_along_axis(rewards[:, i, :], selection, axis=1)[:, 0]

        if i > 0:
            running_count[:, i, :] = running_count[:, i - 1, :]
        update_count = np.zeros((n_iterations, n_machines))
        np.put_along_axis(update_count, selection, 1, axis = 1)
        running_count[:, i, :] = running_count[:, i, :] + update_count

        if i < n_machines and initial_value is None:
            # If initial_value is None, start with initial value observed in machines.
            # NOTE: initial iterations could be randomized, but iterating along machines
            # 1, 2, 3, ... is random enough for this exercise.
            np.put_along_axis(running_value_estimate[:, i, :], selection, strategy_rewards[:, i][:, None], axis=1)
        else:
            if i == 0 and initial_value is not None:
                # If there is an initial_value, start with that.
                running_value_estimate[:, i, :] = initial_value
            else:
                running_value_estimate[:, i, :] = running_value_estimate[:, i - 1, :]

            if alpha is not None:
                # Exponential Weight Decay
                step_size = alpha
            else:
                # Incremental Mean Update
                step_size = 1/np.take_along_axis(running_count[:, i, :], selection, axis=1) 
            
            update_flat = (
                step_size
                * (
                    strategy_rewards[:, i][:, None] 
                    - np.take_along_axis(running_value_estimate[:, i, :], selection, axis=1))
            )
            update = np.zeros((n_iterations, n_machines))
            np.put_along_axis(update, selection, update_flat, axis=1)
            running_value_estimate[:, i, :] = running_value_estimate[:, i, :] + update

    return running_value_estimate, running_count, strategy_rewards, running_selection

exploitation_results = strategy(epsilon=0.0)
average_exploitation_rewards = np.mean(exploitation_results[2], axis=0)
# Plotting
sns.lineplot(x=turns, y=average_exploitation_rewards, color=palette[6], label="exploitation")
sns.lineplot(x=turns, y=average_oracle_rewards, color=palette[4], label="oracle")
sns.lineplot(x=turns, y=average_random_rewards, color=palette[5], label="random")

plt.grid(axis='y', color='#E5E5E5')
sns.despine(left=True, bottom=True, top=True, right=True)
plt.tick_params(axis='x', which='both', bottom=True, left=True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.ylim(0.9, 1.2)
plt.xlabel("turn")
plt.ylabel("average_prize")

formatter = ticker.FuncFormatter(lambda x, pos: f'${x:.2f}')
plt.gca().yaxis.set_major_formatter(formatter)
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
# | label: fig-average_reward_exploration
# | fig-cap: "Running Average of Prizes won by the Exploration Strategy over 5000 iterations"
# | fig-alt: "Line plot of the running Average of prizes won by the Exploration Strategy over 5000 iterations."
exploratation_results = strategy(epsilon=0.2)
average_exploratation_rewards = np.mean(exploratation_results[2], axis=0)
# Plotting
sns.lineplot(x=turns, y=average_exploitation_rewards, color=palette[6], label="exploitation", alpha = 0.3)
sns.lineplot(x=turns, y=average_exploratation_rewards, color=palette[6], label="exploratation")
sns.lineplot(x=turns, y=average_oracle_rewards, color=palette[4], label="oracle")
sns.lineplot(x=turns, y=average_random_rewards, color=palette[5], label="random")

plt.grid(axis='y', color='#E5E5E5')
sns.despine(left=True, bottom=True, top=True, right=True)
plt.tick_params(axis='x', which='both', bottom=True, left=True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.ylim(0.9, 1.2)
plt.xlabel("turn")
plt.ylabel("average_prize")

formatter = ticker.FuncFormatter(lambda x, pos: f'${x:.2f}')
plt.gca().yaxis.set_major_formatter(formatter)
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
#
# | label: fig-average_reward_exp_weighting
# | fig-cap: "Running Average of Prizes won by the Exploration Strategy with exponentially weighted averaging over 5000 iterations"
# | fig-alt: "Line plot of the running Average of prizes won by the Exploration Strategy with exponentially weighted averaging over 5000 iterations."
exp_weighting_results = strategy(epsilon=0.25, alpha=0.15)
average_exp_weighting_rewards = np.mean(exp_weighting_results[2], axis=0)
# Plotting
sns.lineplot(x=turns, y=average_exploitation_rewards, color=palette[6], label="exploitation", alpha=0.3)
sns.lineplot(x=turns, y=average_exploratation_rewards, color=palette[6], label="exploratation", alpha=0.3)
sns.lineplot(x=turns, y=average_exp_weighting_rewards, color=palette[6], label="exp_weighting")
sns.lineplot(x=turns, y=average_oracle_rewards, color=palette[4], label="oracle")
sns.lineplot(x=turns, y=average_random_rewards, color=palette[5], label="random")

plt.grid(axis='y', color='#E5E5E5')
sns.despine(left=True, bottom=True, top=True, right=True)
plt.tick_params(axis='x', which='both', bottom=True, left=True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.ylim(0.9, 1.2)
plt.xlabel("turn")
plt.ylabel("average_prize")

formatter = ticker.FuncFormatter(lambda x, pos: f'${x:.2f}')
plt.gca().yaxis.set_major_formatter(formatter)
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
# | label: fig-average_reward_exp_weighting_optimistic
# | fig-cap: "Running Average of Prizes won by the Exploration Strategy with exponentially weighted averaging and optimistic initial expectations over 5000 iterations"
# | fig-alt: "Line plot of the running Average of prizes won by the Exploration Strategy with exponentially weighted averaging and optimistic initial expectations over 5000 iterations."
optimistic_results = strategy(epsilon=0.25, alpha=0.15, initial_value=1.1)
average_optimistic_rewards = np.mean(optimistic_results[2], axis=0)
# Plotting
sns.lineplot(x=turns, y=average_exploitation_rewards, color=palette[6], label="exploitation", alpha=0.3)
sns.lineplot(x=turns, y=average_exploratation_rewards, color=palette[6], label="exploratation", alpha=0.3)
sns.lineplot(x=turns, y=average_exp_weighting_rewards, color=palette[6], label="exp_weighting", alpha=0.3)
sns.lineplot(x=turns, y=average_optimistic_rewards, color=palette[6], label="optimistic")
sns.lineplot(x=turns, y=average_oracle_rewards, color=palette[4], label="oracle")
sns.lineplot(x=turns, y=average_random_rewards, color=palette[5], label="random")

plt.grid(axis='y', color='#E5E5E5')
sns.despine(left=True, bottom=True, top=True, right=True)
plt.tick_params(axis='x', which='both', bottom=True, left=True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.ylim(0.9, 1.2)
plt.xlabel("turn")
plt.ylabel("average_prize")

formatter = ticker.FuncFormatter(lambda x, pos: f'${x:.2f}')
plt.gca().yaxis.set_major_formatter(formatter)
plt.show()
#
#
#
#
#
#
#
# | label: fig-cdf_prizes
# | fig-cap: "Cumulative Density Function for 5000 iterations of Exploration Strategy with exponentially weighted averaging and optimistic initial expectations"
# | fig-alt: "Cumulative Density Function for 5000 iterations of Exploration Strategy with exponentially weighted averaging and optimistic initial expectations."
def final_stats(rewards: np.array, initial_holding: float=100, cost_per_play: float=1.0):
    holdings = np.cumsum(rewards - np.ones(rewards.shape[1])[None], axis=1) + initial_holding 
    final_holdings = holdings[:, -1]
    final_holdings[np.any(holdings<=0, axis=1)] = 0

    sorted_final_holdings = np.sort(final_holdings)
    cdf_prob = np.arange(1, len(sorted_final_holdings) + 1) / len(sorted_final_holdings)
    return sorted_final_holdings, cdf_prob

stats_oracle = final_stats(oracle_rewards)
stats_random = final_stats(random_rewards)
stats_optimistic = final_stats(optimistic_results[2])

plt.axvline(x=100, color=palette[0])
sns.lineplot(x=stats_oracle[0], y=stats_oracle[1], color=palette[4], label="oracle")
sns.lineplot(x=stats_random[0], y=stats_random[1], color=palette[5], label="random")
sns.lineplot(x=stats_optimistic[0], y=stats_optimistic[1], color=palette[6], label="optimistic")

plt.grid(axis='y', color='#E5E5E5')
sns.despine(left=True, bottom=True, top=True, right=True)
plt.tick_params(axis='x', which='both', bottom=True, left=True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.ylim(0, 1)
plt.xlabel("final_prize_money")
plt.ylabel("continuous_density_function")

plt.annotate("breakeven", (100, 0.1), (300, 0.18),
    arrowprops=dict(facecolor=palette[0], edgecolor=palette[0],
        arrowstyle="simple,tail_width=0.07,head_width=0.7,head_length=1"))
formatter_perc = ticker.FuncFormatter(lambda x, pos: f'{x:.0%}')
formatter_dollar = ticker.FuncFormatter(lambda x, pos: f'${x:.0f}')
plt.gca().yaxis.set_major_formatter(formatter_perc)
plt.gca().xaxis.set_major_formatter(formatter_dollar)
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
