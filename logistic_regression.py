__author__ = 'twoer2'
__email__ = 'dtcf@163.com'

import numpy as np

class LogisticRegressor:
    """Logistic regression classifier

    hat_p = h_theta(x) = sigma(theta.T.dot(x)), where sigma is a
    sigmoid function:
        sigma(t) = 1 / (1 + exp(-t))
    and the binary prediction target label hat_y is:
        hat_y = 0 if hat_p < 0.5 else 1

    cost function J:
        J(theta) = -1/m * sum_{i=1}^m[y^i*log(hat_p^i) + (1-y^i)*log(1-hat_p^i)]
        (partial derivative of theta_j)
        J'(theta) = 1/m * sum_{i=1}^m(sigma(theta.T.dot(x^i))-y^i).dot(x_j^i)
    """
    def __init__(self, learning_rate=1e-3, n_epoch=2000):
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch

    def _sigma(self, t):
        return 1/(1+np.exp(-t))

    def _cost(self, X, y):
        hat_p = self._sigma(X.dot(self.theta))
        sum = 0
        for pred_y, real_y in zip(hat_p, y):
            sum += real_y*np.log(pred_y) + (1-real_y)*np.log(1-pred_y)
        return -sum / X.shape[0]

    def fit(self, X, y):
        # add x_0 = 1 for samples
        X_b = np.insert(X, 0, 1, axis=1)
        # theta: randomly initialized: [-1/sqrt(n), 1/sqrt(n)]
        limit = np.sqrt(X_b.shape[1])
        self.theta = np.random.uniform(-1/limit, 1/limit, (X_b.shape[1], 1))

        for epoch in range(self.n_epoch):
            self.theta -= self.learning_rate \
                          * 1/X_b.shape[0] * X_b.T.dot(self._sigma(X_b.dot(self.theta))-y)

    def predict(self, X):
        X_b = np.insert(X, 0, 1, axis=1)
        return np.around(self._sigma(X_b.dot(self.theta))).astype(int)

    def predict_proba(self, X):
        X_b = np.insert(X, 0, 1, axis=1)
        return [[x[0], y[0]] for x, y in \
                zip(self._sigma(X_b.dot(self.theta)), 1-self._sigma(X_b.dot(self.theta)))]