import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import uniform_filter1d
from pathlib import Path

# Parameters for 1D plot
BETA_1_1d = 0
NUM_OBS_1d = 10000  # Large number of observations to approximate the population SSR
rng_1d = np.random.default_rng(1)

# Generate data for 1D plot
X1_1d = rng_1d.normal(size=NUM_OBS_1d)
Y_1d = BETA_1_1d * X1_1d + 0.5 * rng_1d.normal(size=NUM_OBS_1d)

# Define the SSR function for 1D plot
def ssr_1d(b1, X1, Y):
    return np.mean((Y - b1 * X1) ** 2)

# Generate a grid of beta values for the SSR plot for 1D plot
ref_b_val_1d = 1.5
b1_vals_1d = np.linspace(-ref_b_val_1d, ref_b_val_1d, 1000)
SSR_vals_population_1d = np.array([ssr_1d(b1, X1_1d, Y_1d) for b1 in b1_vals_1d])

# Generate a sequence of growing samples for 1D plot
sample_sizes_1d = np.concatenate(
    [
        np.arange(5, 101),  # Every number between 1 and 100
        np.arange(100, 201, 4),  # Every 2nd number between 100 and 200
        np.arange(200, 501, 5),  # Every 3rd number between 200 and 500
        np.arange(500, 1001, 8),  # Every 5th number between 500 and 1000
    ]
).astype(int)

# Initialize the SSR values for the moving average for 1D plot
SSR_vals_sample_history_1d = []

# Parameters for 2D plot
BETA_1_2d = 0
BETA_2_2d = 0
NUM_OBS_2d = 10000  # Large number of observations to approximate the population SSR
rng_2d = np.random.default_rng(1)

# Generate data for 2D plot
X1_2d = rng_2d.normal(size=NUM_OBS_2d)
X2_2d = rng_2d.normal(size=NUM_OBS_2d)
Y_2d = BETA_1_2d * X1_2d + BETA_2_2d * X2_2d + 0.5 * rng_2d.normal(size=NUM_OBS_2d)

# Define the SSR function for 2D plot
def ssr_2d(b1, b2, X1, X2, Y):
    return np.mean((Y - (b1 * X1 + b2 * X2)) ** 2)

# Generate a grid of beta values for the contour plot for 2D plot
ref_b_val_2d = 0.3
b1_vals_2d = np.linspace(-ref_b_val_2d, ref_b_val_2d, 100)
b2_vals_2d = np.linspace(-ref_b_val_2d, ref_b_val_2d, 100)
B1_2d, B2_2d = np.meshgrid(b1_vals_2d, b2_vals_2d)
SSR_vals_population_2d = np.array([[ssr_2d(b1, b2, X1_2d, X2_2d, Y_2d) for b1 in b1_vals_2d] for b2 in b2_vals_2d])

# Generate a sequence of growing samples for 2D plot
sample_sizes_2d = np.concatenate(
    [
        np.arange(5, 100),  # Every number between 1 and 100
        np.arange(100, 200, 4),  # Every 2nd number between 100 and 200
        np.arange(200, 500, 5),  # Every 3rd number between 200 and 500
        np.arange(500, 1000, 8),  # Every 5th number between 500 and 1000
        np.arange(1000, 2001, 16),  # Every 5th number between 500 and 1000
    ]
).astype(int)

# Initialize the OLS estimator path for 2D plot
ols_path_2d = []

# Set up the figure and the axes using gridspec
fig = plt.figure(figsize=(14, 12))
gs = GridSpec(3, 2, figure=fig, height_ratios=[0.1, 1, 1])
ax_text = fig.add_subplot(gs[0, :])
ax_1d = fig.add_subplot(gs[1, :])
ax_2d_1 = fig.add_subplot(gs[2, 0])
ax_2d_2 = fig.add_subplot(gs[2, 1])

# Remove axes for the text area
ax_text.axis('off')

# Create a color map for the fading effect for 2D plot
cmap_2d = plt.get_cmap('viridis')
norm_2d = Normalize(vmin=0, vmax=len(sample_sizes_2d))
sm_2d = ScalarMappable(norm=norm_2d, cmap=cmap_2d)
sm_2d.set_array([])

# Initialize the SSR values for the moving average for 2D plot
SSR_vals_sample_history_2d = []

# Animation function: this is called sequentially
def animate(i):
    ax_1d.clear()
    ax_2d_1.clear()
    ax_2d_2.clear()

    # 1D Plot
    ax_1d.plot(
        b1_vals_1d,
        SSR_vals_population_1d,
        color="teal",
        label="Population Objective Function",
    )

    n_1d = sample_sizes_1d[i]
    X1_sample_1d = X1_1d[:n_1d]
    Y_sample_1d = Y_1d[:n_1d]

    # Compute the sample SSR function for 1D plot
    SSR_vals_sample_1d = np.array([ssr_1d(b1, X1_sample_1d, Y_sample_1d) for b1 in b1_vals_1d])

    # Apply moving average to smooth the SSR values for 1D plot
    if i == 0:
        SSR_vals_sample_smoothed_1d = SSR_vals_sample_1d
    else:
        SSR_vals_sample_history_1d.append(SSR_vals_sample_1d)
        SSR_vals_sample_smoothed_1d = uniform_filter1d(
            np.array(SSR_vals_sample_history_1d),
            size=min(5, len(SSR_vals_sample_history_1d)),
            axis=0,
            mode="reflect",
        )[-1]

    # Plot the sample SSR function for 1D plot
    ax_1d.plot(
        b1_vals_1d,
        SSR_vals_sample_smoothed_1d,
        color="darkorange",
        label=f"Sample Objective Function (Sample Size={n_1d})",
    )

    # Find the minimizer of the sample SSR for 1D plot
    min_idx_1d = np.argmin(SSR_vals_sample_smoothed_1d)
    b1_min_1d = b1_vals_1d[min_idx_1d]
    ssr_min_1d = SSR_vals_sample_smoothed_1d[min_idx_1d]

    # Plot the broken line from the minimizer to the x-axis for 1D plot
    ax_1d.plot([b1_min_1d, b1_min_1d], [0, ssr_min_1d], "k--")
    ax_1d.text(
        b1_min_1d, 0, r"$\hat{\beta}$", ha="center", va="top", fontsize=12, color="black"
    )

    # Find the minimizer of the population SSR for 1D plot
    pop_min_idx_1d = np.argmin(SSR_vals_population_1d)
    pop_b1_min_1d = b1_vals_1d[pop_min_idx_1d]
    pop_ssr_min_1d = SSR_vals_population_1d[pop_min_idx_1d]

    # Plot the pale broken line from the minimizer of the population SSR to the x-axis for 1D plot
    ax_1d.plot([pop_b1_min_1d, pop_b1_min_1d], [0, pop_ssr_min_1d], "k--", alpha=0.3)

    ax_1d.set_xlim(b1_vals_1d[[0, -1]])
    ax_1d.set_ylim(0, np.max(SSR_vals_population_1d))
    ax_1d.set_xlabel("$b_1$", loc="right")
    ax_1d.set_ylabel("Sum of squared residuals")
    ax_1d.legend(loc="upper center")

    # Remove x and y tick labels for 1D plot
    ax_1d.set_xticks([])
    ax_1d.set_yticks([])

    # 2D Plots
    # Left subplot: Population SSR contours and OLS path for 2D plot
    contour1 = ax_2d_1.contour(
        B1_2d,
        B2_2d,
        SSR_vals_population_2d,
        levels=np.linspace(0, 0.4, 20),
        cmap='cividis',
    )
    ax_2d_1.plot(BETA_1_2d, BETA_2_2d, "ro", label="True Value")  # True value
    ax_2d_1.set_xlim(b1_vals_2d[[0, -1]])
    ax_2d_1.set_ylim(b2_vals_2d[[0, -1]])
    ax_2d_1.set_xlabel('β1')
    ax_2d_1.set_ylabel('β2')
    ax_2d_1.set_title('OLS Estimator vs. Population Objective Function')

    n_2d = sample_sizes_2d[i]
    X1_sample_2d = X1_2d[:n_2d]
    X2_sample_2d = X2_2d[:n_2d]
    Y_sample_2d = Y_2d[:n_2d]

    # Compute the OLS estimator for 2D plot
    X_2d = np.column_stack((X1_sample_2d, X2_sample_2d))
    b_ols_2d = np.linalg.lstsq(X_2d, Y_sample_2d, rcond=None)[0]

    # Update the OLS path for 2D plot
    ols_path_2d.append(b_ols_2d)
    ols_path_array_2d = np.array(ols_path_2d)

    # Plot the OLS path with fading colors for 2D plot
    for j in range(len(ols_path_array_2d)):
        ax_2d_1.plot(ols_path_array_2d[j:j+2, 0], ols_path_array_2d[j:j+2, 1], color=cmap_2d(norm_2d(j)), alpha=0.6)

    ax_2d_1.plot(b_ols_2d[0], b_ols_2d[1], "go")  # Current OLS estimator point
    ax_2d_1.legend()

    # Right subplot: Current sample SSR contours and true value for 2D plot
    SSR_vals_sample_2d = np.array([[ssr_2d(b1, b2, X1_sample_2d, X2_sample_2d, Y_sample_2d) for b1 in b1_vals_2d] for b2 in b2_vals_2d])

    # Apply moving average to smooth the SSR values for 2D plot
    if i == 0:
        SSR_vals_sample_smoothed_2d = SSR_vals_sample_2d
    else:
        SSR_vals_sample_history_2d.append(SSR_vals_sample_2d)
        SSR_vals_sample_smoothed_2d = np.array(SSR_vals_sample_history_2d)
        for dim in range(2):
            SSR_vals_sample_smoothed_2d = uniform_filter1d(
                SSR_vals_sample_smoothed_2d,
                size=min(5, len(SSR_vals_sample_history_2d)),
                axis=dim+1,
                mode='reflect'
            )
        SSR_vals_sample_smoothed_2d = SSR_vals_sample_smoothed_2d[-1]

    contour2 = ax_2d_2.contour(
        B1_2d,
        B2_2d,
        SSR_vals_sample_smoothed_2d,
        levels=np.linspace(0, 0.4, 20),
        cmap='cividis',
    )
    ax_2d_2.plot(BETA_1_2d, BETA_2_2d, "ro", label="True Value")  # True value
    ax_2d_2.set_xlim(b1_vals_2d[[0, -1]])
    ax_2d_2.set_ylim(b2_vals_2d[[0, -1]])
    ax_2d_2.set_xlabel('β1')
    ax_2d_2.set_ylabel('β2')
    ax_2d_2.set_title('Target Value vs. Sample Objective Function')
    ax_2d_2.legend()

# Call the animator
WriterMP4 = animation.writers["ffmpeg"]
writer_mp4 = WriterMP4(fps=20, metadata=dict(artist="bww"), bitrate=1800)
ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(sample_sizes_1d),
    repeat=True,
)
ani.save(Path("images").resolve() / "combined.mp4", writer=writer_mp4)
plt.show()
