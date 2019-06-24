import numpy as np


class Perceptron:
    """binary Perceptron model.

    loss function: -\sum_{x_i misclassified} y_i*(w*x_i+b). sum of distance from the
                   misclassified samples to the hyperplane
    """
    def __init__(self, learning_rate=1e-3, n_epoch=-1):
        """
        :param learning_rate:
        :param n_epoch: if n_epoch == -1, the training process would not be terminated
                        until there is no misclassified sample
        """
        self.learning_rate = learning_rate
        if n_epoch == -1:
            self.n_epoch = 0x7fffffff

    def fit(self, X, y):
        """
        X: n_sample*n_feature, y: n_sample*1
        """
        # add x_0 = 1 for samples
        X_b = np.insert(X, 0, 1, axis=1)
        # theta: randomly initialized: [-1/sqrt(n), 1/sqrt(n)]
        limit = np.sqrt(X_b.shape[1])
        self.theta = np.random.uniform(-1 / limit, 1 / limit, (X_b.shape[1], 1))

        for epoch in range(self.n_epoch):
            # collect misclassified samples in each iteration
            pred = np.sign(X_b.dot(self.theta)).flatten()
            X_b_mis = X_b[pred != y]
            if len(X_b_mis) == 0:
                # no misclassified samples any more
                break
            y_mis = y[pred != y]
            print('loss:', self._loss(X_b_mis, y_mis))
            # update theta
            for i in range(len(X_b_mis)):
                self.theta += self.learning_rate * y_mis[i] * np.expand_dims(X_b_mis[i], axis=1)

    def predict(self, X):
        X_b = np.insert(X, 0, 1, axis=1)
        return np.sign(X_b.dot(self.theta)).flatten()

    def _loss(self, X_mis, y_mis):
        return -np.sum(y_mis * (X_mis.dot(self.theta).flatten()))
