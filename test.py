# -*- endoding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def mse(pred_y, y):
    return np.linalg.norm(pred_y-y)

def diff(pred_y, y):
    return np.sum(pred_y.flatten()!=y.flatten())

def test_linear_regression():
    # from RegularizedLinearRegression import *
    # from LinearRegression import *
    from PolynomialRegression import PolynomialRegressor
    from sklearn.linear_model import LinearRegression
    # X = 2 * np.random.uniform(0, 1, (100, 1))
    X = np.random.rand(100, 1)
    y = 4 + 3 * X # + np.random.randn(100, 1)

    X = np.c_[np.random.uniform(0, 1, (100, 1)), np.random.uniform(0, 5000, (100, 1)), np.random.uniform(0, 100, (100, 1))]
    y = 4 + X.dot(np.array([[300],[2000],[100]]))

    #X = np.c_[np.random.uniform(0, 1, (100, 1)), np.random.uniform(0, 5000, (100, 1)), np.random.uniform(0, 100, (100, 1))]
    #y = 4 + X.dot(np.array([[300],[2000],[100]]))

    lr = PolynomialRegressor(learning_rate=0.001, n_epoch=3000, degree=3)
    lr.fit(X, y)
    test_X = np.c_[np.random.uniform(0, 1, (100, 1)), np.random.uniform(0, 50, (100, 1)), np.random.uniform(0, 100, (100, 1))]
    #test_X = np.random.rand(100, 1)
    test_y = 4 + test_X.dot(np.array([[300],[2000],[100]]))
    #test_y = 4 + 3 * test_X # + np.random.rand(100, 1)
    pred_y = lr.predict(test_X)

    print(lr.theta)
    print(mse(pred_y, test_y))

    # slr = LinearRegression()
    # slr.fit(X, y)
    # print(mse(slr.predict(test_X), test_y))
    # print(slr.intercept_, slr.coef_)

    '''
    we can know that by setting parameter debug=True, stochastic and mini-batch gradient
    descent cost much less epochs to reach a relatively small norm of gradient vector, which
    indicates that stochastic and mini-batch train faster than batch gradient descent.
    '''
    
    # plt.plot(test_X, pred_y, 'r-')
    # plt.plot(X, y, 'b,')
    # plt.axis([0, 2, 0, 15])
    # plt.savefig('linear_regression.png')

def test_logistic_regression():
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris['data'][:, 3:]
    y = (iris['target']==2).astype(np.int).reshape(-1, 1)

    print(X.shape, y.shape)

    from logistic_regression import LogisticRegressor

    lr = LogisticRegressor(learning_rate=0.3, n_epoch=50000)
    lr.fit(X, y)
    # print('predict:', lr.predict_proba(X[:10]))
    # print('target:', y[:10])
    print(diff(lr.predict(X), y))
    print(X[lr.predict(X).flatten()!=y.flatten()])

    from sklearn.linear_model import LogisticRegression

    slr = LogisticRegression()
    slr.fit(X, y.flatten())
    # print(slr.predict_proba(X[:10]))
    print(diff(slr.predict(X), y))

    print(X[slr.predict(X)!=y.flatten()])

def test_softmax_regression():
    from softmax_regression import SoftmaxRegressor
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris['data'][:, (2,3)]
    y = iris['target']

    sr = SoftmaxRegressor(learning_rate=0.3, n_epoch=50000)
    sr.fit(X, y)
    pred = sr.predict(X)
    print('predict:', pred.flatten())
    print('real:', y.flatten())
    print('proba:', sr.predict_proba(X))
    print(diff(pred, y))

if __name__ == '__main__':
    test_softmax_regression()