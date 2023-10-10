import numpy as np
import matplotlib.pyplot as plt

# Define the decay constant
decay_constant = 0.1

# Define the initial condition
Num_atoms = 1000

# Define simulation parameters
a = 0                  # start time
b = 10                 # end time
num_steps = 5         # number of time steps
h = (b - a) / num_steps  # time step size
time_list = np.arange(a, b, h)

# Euler's method
Num_atoms_euler_list = []
for time in time_list:
    Num_atoms_euler_list.append(Num_atoms)
    Num_atoms += h * (-decay_constant * Num_atoms)

# Reset initial condition
Num_atoms = 1000

# 4th Order Runge-Kutta method
Num_atoms_rk_list = []
for time in time_list:
    Num_atoms_rk_list.append(Num_atoms)
    k1 = h * (-decay_constant * Num_atoms)
    k2 = h * (-decay_constant * (Num_atoms + 0.5 * k1))
    k3 = h * (-decay_constant * (Num_atoms + 0.5 * k2))
    k4 = h * (-decay_constant * (Num_atoms + k3))
    Num_atoms += (k1 + 2*k2 + 2*k3 + k4) / 6

# Analytic solution
def analytic_solution(time):
    return 1000 * np.exp(-decay_constant * time)

# Generate analytic solution values
Num_atoms_analytic_list = [analytic_solution(time) for time in time_list]

# Print the final result
print(f'Final Result using Euler\'s Method: {Num_atoms_euler_list[-1]}')
print(f'Final Result using 4th Order Runge-Kutta: {Num_atoms_rk_list[-1]}')
print(f'Final Result using Analytic Solution: {Num_atoms_analytic_list[-1]}')

# Plotting
plt.plot(time_list, Num_atoms_analytic_list, label="Analytic Solution")
plt.scatter(time_list, Num_atoms_euler_list, label="Euler's Method")
plt.scatter(time_list, Num_atoms_rk_list, label="4th Order Runge-Kutta")
plt.xlabel("Time")
plt.ylabel("Number of Atoms")
plt.legend()
plt.show()







