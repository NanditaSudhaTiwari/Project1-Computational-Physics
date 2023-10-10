import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps, trapz

# Define constants
E_rate = 3.8e26  # Energy production rate of Sun in Watts per cubic meter
R = 6.96e8  # Radius of the Sun in meters

# Define the integrable function for luminosity
def luminosity_integrand(r):
    return 4 * np.pi * E_rate * r**2

# Define the limits of integration
a = 0
b = R

# Define the number of subintervals
n = 10000

# Riemann sum
def riemann_sum(func, a, b, n):
    h = (b - a) / n
    integral = 0
    for i in range(n):
        xi = a + i * h
        integral += func(xi) * h
    return integral

# Trapezoidal rule
def trapezoidal_rule(func, a, b, n):
    x = np.linspace(a, b, n+1)
    y = func(x)
    integral = trapz(y, x)
    return integral

# Simpson's rule
def simpsons_rule(func, a, b, n):
    x = np.linspace(a, b, n+1)
    y = func(x)
    integral = simps(y, x)
    return integral

# Analytic Solution
analytic_result = (4/3) * np.pi * E_rate * R**3

# Calculate the luminosity using each method
riemann_luminosity = riemann_sum(luminosity_integrand, a, b, n)
trapezoidal_luminosity = trapezoidal_rule(luminosity_integrand, a, b, n)
simpsons_luminosity = simpsons_rule(luminosity_integrand, a, b, n)

# SciPy implementations
scipy_trapezoidal_luminosity = trapz(luminosity_integrand(np.linspace(a, b, n+1)), np.linspace(a, b, n+1))
scipy_simpsons_luminosity = simps(luminosity_integrand(np.linspace(a, b, n+1)), np.linspace(a, b, n+1))

# Print the results
print(f'Analytic Luminosity: {analytic_result} Watts')
print(f'Riemann Sum Luminosity: {riemann_luminosity} Watts')
print(f'Trapezoidal Rule Luminosity: {trapezoidal_luminosity} Watts (Scipy: {trapezoidal_rule(luminosity_integrand, a, b, n)})')
print(f'Simpson\'s Rule Luminosity: {simpsons_luminosity} Watts (Scipy: {simpsons_rule(luminosity_integrand, a, b, n)})')
print(f"Scipy Luminosity (Trapezoidal Rule): {scipy_trapezoidal_luminosity} Watts")
print(f"Scipy Luminosity (Simpson's Rule): {scipy_simpsons_luminosity} Watts")

# Generate Riemann Sum Plot
riemann_r_values = np.linspace(a, b, n)
riemann_luminosity_values = luminosity_integrand(riemann_r_values)

plt.figure(figsize=(10, 5))
plt.plot(riemann_r_values, riemann_luminosity_values, label='Riemann Sum')
plt.xlabel('Radius (m)')
plt.ylabel('Luminosity Integrand')
plt.title('Riemann Sum vs. Radius')
plt.legend()
plt.grid(True)
plt.show()

# Generate Trapezoidal Rule Plot
trapezoidal_r_values = np.linspace(a, b, n+1)
trapezoidal_luminosity_values = luminosity_integrand(trapezoidal_r_values)

plt.figure(figsize=(10, 5))
plt.plot(trapezoidal_r_values, trapezoidal_luminosity_values, label='Trapezoidal Rule')
plt.xlabel('Radius (m)')
plt.ylabel('Luminosity Integrand')
plt.title('Trapezoidal Rule vs. Radius')
plt.legend()
plt.grid(True)
plt.show()

# Generate Simpson's Rule Plot
simpsons_r_values = np.linspace(a, b, n+1)
simpsons_luminosity_values = luminosity_integrand(simpsons_r_values)

plt.figure(figsize=(10, 5))
plt.plot(simpsons_r_values, simpsons_luminosity_values, label="Simpson's Rule")
plt.xlabel('Radius (m)')
plt.ylabel('Luminosity Integrand')
plt.title("Simpson's Rule vs. Radius")
plt.legend()
plt.grid(True)
plt.show()

# Plot the luminosity function
r_values = np.linspace(a, b, n+1)
luminosity_values = luminosity_integrand(r_values)

plt.figure(figsize=(10, 5))
plt.plot(r_values, luminosity_values, label='Luminosity Integrand')
plt.xlabel('Radius (m)')
plt.ylabel('Luminosity Integrand')
plt.title('Luminosity Integrand vs. Radius')
plt.legend()
plt.grid(True)
plt.show()

# Comparing the results
print("\nComparing Results:")
print(f'Relative Error (Riemann vs Analytic): {abs((riemann_luminosity - analytic_result)/analytic_result)*100:.2f}%')
print(f'Relative Error (Trapezoidal vs Analytic): {abs((trapezoidal_luminosity - analytic_result)/analytic_result)*100:.2f}%')
print(f'Relative Error (Simpson\'s vs Analytic): {abs((simpsons_luminosity - analytic_result)/analytic_result)*100:.2f}%')

