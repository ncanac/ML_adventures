"""

This script implements binary logistic regression to the classification
problem in the second part of exercise 2 of Andrew Ng's machine learning
course on Coursera.

"""

import numpy as np
from numpy.polynomial.polynomial import polyval
import matplotlib.pyplot as plt
from sklearn import linear_model as lm

def mapFeatures(x1, x2, deg):
    """
    Takes two features as inputs and maps them to a set of polynomial
    features up to degree specified by deg.

    The features are comprised of:
    x1*x2, x1^2, x2^2, x1^2*x2, x1*x2^2, ... , x1^deg, x2^deg

    These are returned in a new feature matrix X.
    """
    # Create an empty feature matrix to hold all the polynomial features
    X = np.zeros((len(x1), sum(range(2, deg+2))))
    col = 0
    for i in range(1, deg+1):
        for j in range(0, i+1):
            X[:, col] = x1**(i-j) + x2**j
            col += 1
    return X

# Load the data from ex2data2.txt into X and y
data = np.loadtxt("ex2data2.txt")
X = data[:, 0:2]
y = data[:, 2]

# Add polynomial features up to degree 6 to allow for a nonlinear decision boundary
degree = 6
Xpoly = mapFeatures(X[:, 0], X[:, 1], degree)

# Create a LogisticRegression object
logreg = lm.LogisticRegression(penalty='l2', dual=False, C=1, solver='lbfgs')

# Fit the data
logreg.fit(Xpoly, y)

# Plot decision boundary
x1min, x1max = np.min(X[:, 0]), np.max(X[:, 0])
x2min, x2max = np.min(X[:, 1]), np.max(X[:, 1]) 
x1pad, x2pad = 0.1*(x1max - x1min), 0.1*(x2max - x2min)
h = 0.02*x1pad
xx, yy = np.meshgrid(np.arange(x1min-x1pad, x1max+x1pad+h, h), np.arange(x2min-x2pad, x2max+x2pad+h, h))
Xgrid = mapFeatures(xx.ravel(), yy.ravel(), degree)
probs = logreg.predict_proba(Xgrid)[:, 1].reshape(xx.shape)
f, ax = plt.subplots(figsize=(10, 7.5))
contour = ax.contourf(xx, yy, probs, 50, cmap="RdBu", vmin=0, vmax=1)

# Print some predictions
x1test = np.array([-0.619])
x2test = np.array([0.221])

ax_c = f.colorbar(contour)
ax_c.set_label("$P(y = 1)$", size=20)
ax_c.set_ticks([0, .25, .5, .75, 1])

# Plot the data, positive values in blue and negative values in red
ax.scatter(X[:,0], X[:, 1], c=y, s=50, cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white", linewidth=1)

plt.xlim(x1min - x1pad, x1max + x1pad)
plt.ylim(x2min - x2pad, x2max + x2pad)
plt.xlabel("$x_1$", size=20)
plt.ylabel("$x_2$", size=20)
plt.tight_layout()

plt.show()
#plt.savefig("log_reg_nl.png")
