import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

def fit_poly(x, y):
    x = np.array(x)

    # Fit a 3rd-degree polynomial to the data
    p = Polynomial.fit(x, y, deg=3)

    return p

x = [56, 62, 63, 72, 77, 79, 93, 102, 106, 116, 122, 149, 161]
y = np.array([3.6, 3.778, 3.872, 3.9267, 3.977, 4.168, 4.256, 4.36, 4.434, 4.4726, 4.54, 4.612, 4.661])
y = 5.354397 - y

print("x: ", x)
print("y: ", y)
p = fit_poly(x, y)
print(p)

# Plot the data and the fitted polynomial
plt.plot(x, y, 'o')
# plt.plot(p, '-')
plt.show()
#0.92523112 - 0.42423197·x + 0.28328111·x² - 0.08428235·x³