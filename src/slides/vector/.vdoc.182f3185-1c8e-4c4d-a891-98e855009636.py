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
#| echo: true
#| code-fold: true
#| code-summary: "Expand for full data preparation code"
import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.regression.linear_model import OLS

# Read in the data
data_path = ("https://github.com/pegeorge/Econ521_Datasets/"
             "raw/refs/heads/main/cps09mar.csv")
cps_data = pd.read_csv(data_path)

# Generate variables
cps_data["experience"] = cps_data["age"] - cps_data["education"] - 6
cps_data["experience_sq_div"] = cps_data["experience"]**2/100
cps_data["wage"] = cps_data["earnings"]/(cps_data["week"]*cps_data["hours"] )
cps_data["log_wage"] = np.log(cps_data['wage'])

# Retain only married women white with present spouses
select_data = cps_data.loc[
    (cps_data["marital"] <= 2) & (cps_data["race"] == 1) & (cps_data["female"] == 1), :
]

# Construct X and y for regression 
exog = select_data.loc[:, ['education', 'experience', 'experience_sq_div']]
exog = sm.add_constant(exog)
endog = select_data.loc[:, "log_wage"]
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
results = OLS(endog, exog).fit(cov_type='HC0')
print(results.summary())
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
#| eval: true 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import beta, norm, nct
from scipy.ndimage import uniform_filter1d
from statsmodels.distributions.empirical_distribution import ECDF
from pathlib import Path

# Function to generate samples from a given distribution
def generate_samples(distribution, params, size):
    return distribution.rvs(*params, size=size)

# Parameters for the Beta distribution
beta_params = (0.5, 0.5)

# Parameters for the noncentral t distribution
nct_params = (3, 10, 0, 1)  # df=3, nc=10, loc=0, scale=1

# Choose the distribution (uncomment the desired distribution)
# distribution = beta
# params = beta_params
distribution = nct
params = nct_params

# Generate sample sizes
sample_sizes = np.concatenate(
    [
        np.arange(5, 101),  # Every number between 1 and 100
        np.arange(100, 201, 4),  # Every 2nd number between 100 and 200
        np.arange(200, 501, 5),  # Every 3rd number between 200 and 500
        np.arange(500, 3001, 8),  # Every 5th number between 500 and 1000
    ]
).astype(int)

# Initialize the history of standardized sample means for smoothing
standardized_means_history = []

# Set up the figure and the axes
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG_COLOR)
fig.patch.set_edgecolor("teal")
fig.patch.set_linewidth(5)


overall_samples = generate_samples(distribution, params, size=(80000, sample_sizes.max()))
population_mean = np.mean(overall_samples)
population_std = np.std(overall_samples)

# Animation function: this is called sequentially
def animate(i):
    ax.clear()

    # Get the current sample size
    n = sample_sizes[i]

    # Draw samples from the chosen distribution
    samples = overall_samples[:, :n]

    # Compute the sample means
    sample_means = np.mean(samples, axis=1)

    # Standardize the sample means
    standardized_means = (sample_means - population_mean) / population_std * np.sqrt(n)

    # Apply moving average to smooth the standardized sample means
    # if i == 0:
    #     smoothed_means = standardized_means
    # else:
    #     standardized_means_history.append(standardized_means)
    #     smoothed_means = uniform_filter1d(
    #         np.array(standardized_means_history),
    #         size=min(5, len(standardized_means_history)),
    #         axis=0,
    #         mode="reflect",
    #     )[-1]

    # Compute the empirical CDF
    ecdf = ECDF( standardized_means)
    x_vals = np.linspace(-4, 4, 1000)
    y_vals = ecdf(x_vals)

    # Plot the empirical CDF
    ax.plot(x_vals, y_vals, label=f'Sample Size = {n}', color='darkorange')

    # Plot the standard normal CDF for reference
    ax.plot(x_vals, norm.cdf(x_vals), label='Standard Normal', color='teal', linestyle='--')

    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Standardized Sample Mean')
    ax.set_ylabel('CDF')
    ax.set_title('Central Limit Theorem - Convergence of CDFs')
    ax.legend(loc='lower right')

# Call the animator
WriterMP4 = animation.writers["ffmpeg"]
writer_mp4 = WriterMP4(fps=20, metadata=dict(artist="bww"), bitrate=1800)
ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(sample_sizes),
    repeat=True,
)
ani.save(Path("images").resolve() / "clt_cdf.mp4", writer=writer_mp4)

plt.close()

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
