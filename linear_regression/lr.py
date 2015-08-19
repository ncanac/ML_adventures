import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt

def main():
    # Read in data:
    data = np.loadtxt('ex1data1.txt')
    m = len(data)
    X = np.ones((m, 2))
    X[:, 1] = data[:, 0] # population size
    y = data[:, 1] # profit

    # Run linear regression:
    lr = linear_model.LinearRegression()
    lr.fit(X, y)

    # Predict profits for populations of 35,000 and 70,000
    predict1 = lr.predict([1, 3.5])[0]
    print "For population = 35000, we predict a profit of ${:.2f}.".format(predict1*10000)
    predict2 = lr.predict([1, 7])[0]
    print "For population = 70000, we predict a profit of ${:.2f}.".format(predict1*10000)

    # Plot data and model
    plt.scatter(X[:, 1], y)
    plt.plot(X[:, 1], lr.predict(X))
    plt.show()

if __name__ == "__main__":
    main()
