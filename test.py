# -*- endoding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from LinearRegression import *

def test_linear_regression():
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    lr = StochasticGradientRegressor()
    lr.fit(X, y)
    test_X = np.array([[0], [2]])
    pred_y = lr.predict(test_X)

    print(lr.theta)
    print(pred_y)

    '''
    we can know that by setting parameter debug=True, stochastic and mini-batch gradient
    descent cost much less epochs to reach a relatively small norm of gradient vector, which
    indicates that stochastic and mini-batch train faster than batch gradient descent.
    '''
    
    # plt.plot(test_X, pred_y, 'r-')
    # plt.plot(X, y, 'b,')
    # plt.axis([0, 2, 0, 15])
    # plt.savefig('linear_regression.png')

if __name__ == '__main__':
    test_linear_regression()