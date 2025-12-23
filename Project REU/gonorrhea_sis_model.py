import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import savgol_filter
import pysindy as ps

# Realistic parameters for gonorrhea (per day)
beta = 0.2    # transmission rate per day
gamma = 0.1   # recovery rate per day (~10-day infectious period)

# SIS model ODE
def sis_model(I, t, beta, gamma):
    return beta * I * (1 - I) - gamma * I

# Time vector
t = np.linspace(0, 50, 500)
dt = t[1] - t[0]
I0 = 0.01  # initial infected proportion

# Simulate "true" data (no noise)
I_true = odeint(sis_model, I0, t, args=(beta, gamma)).flatten()

# Different noise scenarios
noise_levels = [0.0, 0.001, 0.01]  # none, mild, heavy
labels = ['No Noise', 'Mild Noise', 'Heavy Noise']
colors = ['k', 'b', 'r']

# Savitzky-Golay parameters (for smoothing and derivative)
window_length = 51  # large window to reduce noise effects
polyorder = 3

# Run for each scenario
plt.figure(figsize=(10, 6))
for noise_level, label, color in zip(noise_levels, labels, colors):
    # Add noise
    I_noisy = I_true + noise_level * np.random.randn(len(I_true))

    # Smooth and compute derivative
    I_smooth = savgol_filter(I_noisy, window_length, polyorder)
    dI_dt = savgol_filter(I_noisy, window_length, polyorder, deriv=1, delta=dt)

    # Prepare data for SINDy
    X = I_noisy.reshape(-1, 1)
    X_dot = dI_dt.reshape(-1, 1)

    # Fit SINDy (lower threshold for sensitivity)
    model = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=2),
        optimizer=ps.STLSQ(threshold=0.005)
    )
    model.fit(X, t=dt, x_dot=X_dot)
    print(f"\n{label} SINDy Discovered Model:")
    model.print()

    # Simulate from learned model
    I_pred = model.simulate(np.array([I0]), t)

    # Plot results
    plt.plot(t, I_pred, linestyle='--', color=color, label=f'{label} (SINDy)')
    if noise_level == 0.0:
        plt.plot(t, I_true, 'g-', label='True Dynamics')

# Final plot settings
plt.xlabel('Time (days)')
plt.ylabel('Infected Proportion')
plt.title('SIS Model Recovery with Varying Noise Levels')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
