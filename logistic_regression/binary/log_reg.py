"""

This script implements binary logistic regression to the classification
problem in exercise 2 of Andrew Ng's machine learning course on Coursera.

"""

import numpy as np
from numpy.polynomial.polynomial import polyval
import matplotlib.pyplot as plt
from sklearn import linear_model as lm

# Load the data from ex2data1.txt into X and y
data = np.loadtxt("ex2data1.txt")
X = data[:, 0:2]
y = data[:, 2]

# Create a LogisticRegression object
logreg = lm.LogisticRegression(penalty='l2', dual=False, C=1, solver='lbfgs')

# Fit the data
logreg.fit(X, y)

# Plot decision boundary
x1min, x1max = np.min(X[:, 0]) - 10, np.max(X[:, 0]) + 10
x2min, x2max = np.min(X[:, 1]) - 10, np.max(X[:, 1]) + 10 
d = 0.1
xx, yy = np.mgrid[x1min:x1max:d, x2min:x2max:d]
probs = logreg.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.figure(1, figsize=(8, 6))
plt.pcolormesh(xx, yy, probs, cmap="RdBu")

# Plot the data, positive values in blue and negative values in red
plt.scatter(X[:,0], X[:, 1], c=y, s=50, cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white", linewidth=1)

plt.xlim(x1min, x1max)
plt.ylim(x2min, x2max)
plt.xlabel("$x_1$", size=20)
plt.ylabel("$x_2$", size=20)
plt.tight_layout()

#plt.show()
plt.savefig("log_reg_plot1.png")
