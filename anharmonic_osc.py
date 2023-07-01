# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 23:08:26 2023

@author: prajj
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from matplotlib.backends.backend_pdf import PdfPages

pd.set_option('mode.chained_assignment', None)

# Define the differential equation


def f(t, y, b, A_force, Omega, omega, a5):
    # Forced oscillation term
    a0 = -A_force * np.cos(Omega * t)

    # Array stripping
    x, v = y

    # 1st order ODEs return
    return [
        v,
        -2 * b * v - a0 - omega * omega * x - a5 * (x**5),
    ]


def solver(i, j, b, A_force, Omega_cont, omega, a5):
    sol = solve_ivp(
        f,
        [t0, tf],
        [x0, v0],
        method="LSODA",
        args=(
            b,
            A_force,
            Omega_cont[j],
            omega,
            a5n[i],
        ),
    )

    # Extract the solution
    temp = pd.DataFrame({"t": sol.t, "x": sol.y[0]})
    temp = temp.loc[temp["t"] > 60]
    xmax = np.float64(temp["x"].max())
    temp2 = pd.DataFrame(
        {"a5": a5n[i], "Omega": Omega_cont[j], "amplitude": [xmax]}
    )
    return temp2


# Set initial conditions
x0 = 0  # initial position
v0 = 1  # initial velocity
t0 = 0  # initial time
tf = 100  # final time

b = 0.25  # damping factor
A_force = 1.0  # amplitude of forced oscillation
Omega = 1.0  # driving frequency
omega = 1.0  # natural frequency
dOmega = 0.001  # spacing for driving frequency in continuous case

# Varying the 3rd order term
a5 = np.linspace(0, 0.9, 10)
a5n = np.arange(0,1.1,0.1)

# Create pandas dataframe
df = pd.DataFrame()

for i in tqdm(range(len(a5)),desc="Discrete Omega",ascii="123456789#"):
    # Solve the differential equation using LSODA method
    sol = solve_ivp(
        f,
        [t0, tf],
        [x0, v0],
        method="LSODA",
        args=(
            b,
            A_force,
            Omega,
            omega,
            a5[i],
        ),
    )

    # Extract the solution
    temp = pd.DataFrame({"a5": a5[i], "t": sol.t, "x": sol.y[0], "v": sol.y[1]})
    df = pd.concat([df, temp], ignore_index=True)

# Similar code for plotting resonance profile
df2 = pd.DataFrame()
Omega_cont = np.arange(0.5, 2.0 + dOmega, dOmega)  # continuous-like Omega
for i in tqdm(range(len(a5n)),desc="Continuous Omega",ascii="123456789#"):
    df_temp = Parallel(n_jobs=-1)(
        delayed(solver)(i, j, b, A_force, Omega_cont, omega, a5)
        for j in range(len(Omega_cont))
    )
    df_concat = pd.concat(df_temp, ignore_index=True)
    df2 = pd.concat([df2, df_concat], ignore_index=True)

# Plotting
pp = PdfPages("anharmonic_osc_plots.pdf")

# Page 1
# Create a figure and set its size
fig, axes = plt.subplots(5, 2, figsize=(8.27, 11.69))

# Flatten the axes array to simplify indexing
axes = axes.flatten()

# Loop through the data and create subplots
for i, ax in enumerate(axes):
    temp = df.loc[df["a5"] == a5[i]]
    ax.plot(temp["t"], temp["x"], color="blue")
    ax.set_xlabel("Time")
    ax.set_ylabel("Position")
    ax.set_title(r"$a_5 = $" + "{:.2f}".format(a5[i]))
    ax.grid(True)

plt.suptitle(
    "Forced anharmonic oscillator with varying strength of 6th order anharmonicity in potential\nPosition Plot || $\Omega$="
    + "{:.3f}".format(Omega)
)

# Adjust the spacing between subplots
fig.tight_layout()

# Save the figure to the PDF file
pp.savefig(fig)

# Close the figure
plt.close(fig)

# Page 2
fig, axes = plt.subplots(5, 2, figsize=(8.27, 11.69))

# Flatten the axes array to simplify indexing
axes = axes.flatten()

# Loop through the data and create subplots
for i, ax in enumerate(axes):
    temp = df.loc[df["a5"] == a5[i]]
    ax.plot(temp["t"], temp["v"], color="red")
    ax.set_xlabel("Time")
    ax.set_ylabel("Velocity")
    ax.set_title(r"$a_5 = $" + "{:.2f}".format(a5[i]))
    ax.grid(True)

plt.suptitle(
    "Forced anharmonic oscillator with varying strength of 6th order anharmonicity in potential\nVelocity Plot || $\Omega$="
    + "{:.3f}".format(Omega)
)

# Adjust the spacing between subplots
fig.tight_layout()

# Save the figure to the PDF file
pp.savefig(fig)

# Close the figure
plt.close(fig)
# Page 3
fig, axes = plt.subplots(5, 2, figsize=(8.27, 11.69))

# Flatten the axes array to simplify indexing
axes = axes.flatten()

# Loop through the data and create subplots
for i, ax in enumerate(axes):
    temp = df.loc[df["a5"] == a5[i]]
    ax.plot(temp["x"], temp["v"], color="magenta")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(r"$a_5 = $" + "{:.2f}".format(a5[i]))
    ax.grid(True)

plt.suptitle(
    "Forced anharmonic oscillator with varying strength of 6th order anharmonicity in potential\nPhase Space Plot || $\Omega$="
    + "{:.3f}".format(Omega)
)

# Adjust the spacing between subplots
fig.tight_layout()

# Save the figure to the PDF file
pp.savefig(fig)

# Close the figure
plt.close(fig)

# Page 4
plt.figure(figsize=(8.27, 11.69))

plt.suptitle(
    "Forced anharmonic oscillator with varying strength of 6th order anharmonicity in potential"
)
# x = np.arange(0.75, 1.75, 0.01)
M = []
plt.subplot(3, 1, 1)
for i in range(len(a5n)):
    temp = df2.loc[df2["a5"] == a5n[i]]
    temp.reset_index(inplace=True, drop=True)
    plt.plot(
        temp["Omega"],
        temp["amplitude"],
        label=r"$a_5$=" + "{:.2f}".format(a5n[i]),
    )

plt.xlabel(r"Driving frequency ($\Omega$)")
plt.ylabel("Amplitude")
plt.title("Resonance Profile")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
for i in range(len(a5n)):
    temp = df2.loc[df2["a5"] == a5n[i]]
    temp.is_copy = None
    temp["grad"] = np.gradient(temp["amplitude"], dOmega)
    
    M.append(temp["grad"].min())
    plt.plot(
        temp["Omega"], temp["grad"], label=r"$a_5$=" + "{:.2f}".format(a5n[i])
    )
plt.xlabel(r"Driving frequency ($\Omega$)")
plt.ylabel("Gradient of Amplitude")
plt.title("Gradient of Resonance Profile")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(df2["a5"].unique(), M)
plt.xlabel(r"$a_5$")
plt.ylabel("Min slope")
plt.title(r"Minimum Slope vs $a_5$")
plt.grid(True)
# Adjust the spacing between subplots
plt.tight_layout()

# Save the figure to the PDF file
pp.savefig()

# Close the figure
plt.close()

pp.close()
