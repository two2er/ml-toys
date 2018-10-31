__author__ = 'twoer2'
__email__ = 'dtcf@163.com'

import numpy as np

class SoftmaxRegressor:
    """ softmax regression: a generalization of logistic regression

    http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/

    J(theta) = -[sum_{i=1}^m sum_{k=1}^K 1{y^{(i)}=k}log P(y^{(i)}=k)]
    where P(y^{(i)}=k) = exp(theta^{(k)}.T.dot(x^{(i)})) / sum_{j=1}^k[exp(theta^{(j)}.T.dot(x^{(i)}))]

    J_delta(theta) = -1/m * X^T.dot(ONE-P)
    where ONE is a m*k matrix, ONE_{i, h} is 1{y^{(i)}=h} (i-th sample, h-th class)
          P   is a m*k matrix, P_{i, h} is P(y^{(i)}=h)
    if the shape of the input y is n_sample * n_class (one-hot), then ONE = y
    """
    def __init__(self, learning_rate=1e-3, n_epoch=2000):
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch

    def _P(self, X):
        P = np.exp(X.dot(self.theta))
        normal = np.sum(P, axis=1).reshape(-1, 1)
        return P/normal

    def _one_hot(self, n_sample, y):
        if y.shape == (n_sample, 1) or y.shape == (n_sample,):
            n_class = np.max(y)
            return_y = np.zeros((n_sample, n_class+1))
            for sample, class_ in enumerate(y.reshape(-1)):
                return_y[sample][class_] = 1
            return return_y
        else:
            return y

    def fit(self, X, y):
        """X: n_sample*n_feature, y: n_sample*n_class"""
        # add x_0 = 1 for samples
        X_b = np.insert(X, 0, 1, axis=1)
        y = self._one_hot(X_b.shape[0], y)
        # theta: randomly initialized: [-1/sqrt(n), 1/sqrt(n)]
        limit = np.sqrt(X_b.shape[1])
        # theta: n_features * n_class
        self.theta = np.random.uniform(-1/limit, 1/limit, (X_b.shape[1], y.shape[1]))

        for epoch in range(self.n_epoch):
            self.theta -= self.learning_rate * 1/X_b.shape[0] * X_b.T.dot(self._P(X_b)-y)

    def predict(self, X):
        X_b = np.insert(X, 0, 1, axis=1)
        P = self._P(X_b)
        return np.argmax(P, axis=1).reshape(-1, 1)

    def predict_proba(self, X):
        X_b = np.insert(X, 0, 1, axis=1)
        return self._P(X_b)
        