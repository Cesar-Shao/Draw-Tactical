import numpy as np
import matplotlib.pyplot as plt

def bezier_curve(control_points, num_points=100):
    t = np.linspace(0, 1, num_points)
    
    n = len(control_points) - 1
    curve = np.zeros((num_points, 2))

    for i in range(num_points):
        for j in range(n + 1):
            curve[i] += control_points[j] * binomial_coefficient(n, j) * (1 - t[i])**(n - j) * t[i]**j

    return curve

def binomial_coefficient(n, k):
    return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))

