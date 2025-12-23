
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def generate_si_tcruzi_data(
    alpha=0.33,           # ~1 bite every 3 days
    beta=0.2,             # 20% chance of transmission per bite
    r=0.0167,             # ~60-day progression/recovery time
    Ih0=0.01,
    noise_level=0.01,
    t_end=180,            # Simulate 6 months
    num_points=1000,      # More resolution
    output_csv_path="si_model_t_cruzi_simulation.csv",
    show_plot=True
):

    # Time array
    t = np.linspace(0, t_end, num_points)

    # Time-varying infected vector proportion Iv(t)
    def Iv(t):
        return 0.3 + 0.1 * np.sin(2 * np.pi * t / 30)  # periodic between 0.2 and 0.4

    # SI model with time-varying Iv
    def si_model(Ih, t):
        Iv_t = Iv(t)
        infection_term = (alpha * beta * Iv_t) / (alpha * Iv_t + r)
        return infection_term * (1 - Ih)

    # Solve the ODE
    Ih_true = odeint(si_model, Ih0, t).flatten()

    # Add noise
    Ih_noisy = Ih_true + np.random.normal(0, noise_level, size=Ih_true.shape)
    Ih_noisy = np.clip(Ih_noisy, 0, 1)  # keep within [0,1]

    # Compute Iv values
    Iv_values = Iv(t)

    # Save to CSV
    df = pd.DataFrame({
        "time": t,
        "Ih_true": Ih_true,
        "Ih_noisy": Ih_noisy,
        "Iv": Iv_values
    })
    df.to_csv(output_csv_path, index=False)

    # Optional plot
    if show_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(t, Ih_true, label='True I_h(t)', alpha=0.8)
        plt.plot(t, Ih_noisy, label='Noisy I_h(t)', linestyle='--', alpha=0.7)
        plt.plot(t, Iv_values, label='I_v(t) (vector infection)', color='green', linestyle=':')
        plt.xlabel('Time (days)')
        plt.ylabel('Proportion')
        plt.title('Time-Varying SI Model with Infected Vectors and Noise')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return df, Iv

# Example usage
if __name__ == "__main__":
    df, Iv_function = generate_si_tcruzi_data()
