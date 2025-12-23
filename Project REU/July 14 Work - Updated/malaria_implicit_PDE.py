import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps

# === Load malaria simulation data ===
data = pd.read_csv('malaria_data.csv')
I = data['I(t)'].values.reshape(-1, 1)
dI = data['dI/dt'].values.reshape(-1, 1)
t = data['time'].values

# === Create a polynomial library for 1D time series ===
library = ps.PolynomialLibrary(degree=3, include_bias=False)

# === Initialize and fit the SINDy model ===
model = ps.SINDy(
    feature_library=library,
    optimizer=ps.STLSQ(threshold=1e-5)
)

model.fit(I, t=t, x_dot=dI)

# === Display discovered equation ===
print("\n🔍 Discovered malaria dynamics (explicit form):")
model.print()

# === Simulate the discovered SINDy model ===
I_sindy = model.simulate(I[0], t)

# === Plot side-by-side comparison ===
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Original simulation
axs[0].plot(t, I, label='Original I(t)', color='blue')
axs[0].set_title('Original Malaria ODE Simulation')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Proportion Infected')
axs[0].grid(True)
axs[0].legend()

# SINDy prediction
axs[1].plot(t, I_sindy, '--', label='SINDy Prediction', color='green')
axs[1].set_title('SINDy Model Reconstruction')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Proportion Infected')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()