import numpy as np
from scipy.optimize import curve_fit

sizes = np.load("sizes.npy")
distances = np.load("distances.npy") - 4.459

#sort sizes & change distances to order of sizes
sizes = np.sort(sizes)
distances = distances[np.argsort(sizes)]



print("sizes: ", sizes)
print("distances: ", distances)
print("len(sizes): ", len(sizes))
print("len(distances): ", len(distances))
print(list(zip(distances, sizes)))

def inverse_function(x, a, b):
    return a / x + b

def fit(distances, sizes):
    # Fit the inverse_function to the data (distances and sizes)
    popt, _ = curve_fit(inverse_function, distances, sizes)

    # Print the optimized parameters (a, b) for the fitted function
    print("Optimized parameters (a, b):", popt)

    return popt

# Fit the function and get the optimized parameters
params = fit(distances, sizes)
