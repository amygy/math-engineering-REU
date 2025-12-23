import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
import pysindy as ps

# === Define the SIR model ===
def sir_model(t, y, beta, gamma):
    S, I, R = y
    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I
    return [dS, dI, dR]

# === Parameters ===
beta = 0.3
gamma = 0.1
y0 = [0.99, 0.01, 0.0]  # Initial conditions: 99% susceptible, 1% infected, 0% recovered
t = np.linspace(0, 100, 500)  # Time span

# === Simulate true SIR model ===
sol = solve_ivp(sir_model, [t[0], t[-1]], y0, t_eval=t, args=(beta, gamma))
S, I, R = sol.y

# === Compute numerical derivatives ===
dt = t[1] - t[0]
dS = savgol_filter(S, 15, 3, deriv=1, delta=dt)
dI = savgol_filter(I, 15, 3, deriv=1, delta=dt)
dR = savgol_filter(R, 15, 3, deriv=1, delta=dt)

# === Prepare data for SINDy ===
X = np.vstack((S, I, R)).T
X_dot = np.vstack((dS, dI, dR)).T

# === Define custom library functions as lambdas with separate inputs ===
library_functions = [
    lambda S, I, R: S,        # S
    lambda S, I, R: I,        # I
    lambda S, I, R: R,        # R
    lambda S, I, R: S * I     # S*I interaction term
]

library_function_names = ['S', 'I', 'R', 'S*I']

# Remove function_names from CustomLibrary to avoid errors:
custom_library = ps.CustomLibrary(
    library_functions=library_functions,
    # function_names=['S', 'I', 'R', 'S*I']  # <-- omit this line!
)

# === Define optimizer ===
optimizer = ps.STLSQ(threshold=1e-5)

# Create SINDy model with feature_names:
model = ps.SINDy(feature_library=custom_library,
                 optimizer=optimizer,
                 feature_names=['S', 'I', 'R'])

model.fit(X, t=dt, x_dot=X_dot)

print("\n🔍 Discovered SIR dynamics (SINDy):")
model.print()  # No args here

# === Simulate SINDy model ===
X_sindy = model.simulate(X[0], t)

# === Plot comparison ===
plt.figure(figsize=(12, 6))
plt.plot(t, I, label='True I(t)', color='blue')
plt.plot(t, X_sindy[:, 1], '--', label='SINDy I(t)', color='green')
plt.xlabel('Time')
plt.ylabel('Infected Proportion')
plt.title('SIR Model vs SINDy Reconstruction')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
