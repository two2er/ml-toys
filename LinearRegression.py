# -*- endoding: utf-8 -*-

import numpy as np
import abc

class LinearRegressor:

    def __init__(self, learning_rate = 1e-3, tolerance = 1e-3):
        """
        if the training method is gradient descent, the training process would be ended after the
        l2-norm of gradient vector is smaller than tolerance e.
        """
        self.learning_rate = learning_rate
        self.tolerance = tolerance

    def fit(self, X, y):
        """
        begin to train a model based on trainset X and target value set y.
        if the gradient descent method is mini-batch, the parameter batch_size would be taken as
        the size of one batch
        """

        # add x_0 = 1 for samples
        X_b = np.insert(X, 0, 1, axis=1)
        # theta: randomly initialized: [-1/sqrt(n), 1/sqrt(n)]
        limit = np.sqrt(X_b.shape[1])
        self.theta = np.random.uniform(-1/limit, 1/limit, (X_b.shape[1], 1))

        self.training_method(X_b, y)

    @abc.abstractmethod
    def training_method(self, X, y):
        pass

    def predict(self, X):
        """
        predict the target values of X
        """
        assert self.theta is not None, 'you must train a model before predicting'
        X_b = np.insert(X, 0, 1, axis=1)
        return X_b.dot(self.theta)

class NormalRegressor(LinearRegressor):
    """
    training with the Normal Equation
    hat_theta = (X.T.dot(X)).inverse.dot(X.T).dot(y)
    in fact NormalRegressor should not inherit from LinearRegressor because it does not have
    learning_rate and tolerance attributes.
    """
    def training_method(self, X, y):
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

class BatchGradientRegressor(LinearRegressor):

    def training_method(self, X, y):
        while True:
            gradient_vector = 2 / X.shape[0] * X.T.dot(X.dot(self.theta) - y)
            if np.linalg.norm(gradient_vector) < self.tolerance:
                break
            self.theta = self.theta - self.learning_rate * gradient_vector

class StochasticGradientRegressor(LinearRegressor):

    def training_method(self, X, y):
        m = X.shape[0]
        epoch_count = 0
        while True:
            # evaluate the norm of gradient vector every 10 epochs. or terminate after 2000 epochs
            epoch_count += 1
            if epoch_count % 10 == 0:
                gradient_vector = 2 / m * X.T.dot(X.dot(self.theta) - y)
                if np.linalg.norm(gradient_vector) < self.tolerance:
                    break
            if epoch_count == 2000:
                break
            # randomly shuffle X
            random_sequence = np.random.permutation([i for i in range(m)])
            self.learning_rate = self.learning_schedule(self.learning_rate)
            for i in range(m):
                xi = X[random_sequence[i]:random_sequence[i]+1]
                yi = y[random_sequence[i]:random_sequence[i]+1]
                gradient_vector = 2 * xi.T.dot(xi.dot(self.theta) - yi)
                self.theta = self.theta - self.learning_rate * gradient_vector

    def learning_schedule(self, learning_rate):
        # learning rate decay
        return learning_rate * 0.99
    
class MiniBatchGradientRegressor(LinearRegressor):

    def __init__(self, learning_rate = 1e-3, tolerance = 1e-3, batch_size=20):
        self.batch_size = batch_size
        super(MiniBatchGradientRegressor, self).__init__(learning_rate, tolerance)

    def training_method(self, X, y):
        m = X.shape[0]
        epoch_count = 0
        while True:
            # evaluate the norm of gradient vector every 10 epochs. or terminate after 2000 epochs
            epoch_count += 1
            if epoch_count % 10 == 0:
                gradient_vector = 2 / m * X.T.dot(X.dot(self.theta) - y)
                if np.linalg.norm(gradient_vector) < self.tolerance:
                    break
            if epoch_count == 2000:
                break
            # randomly shuffle X
            random_sequence = np.random.permutation([i for i in range(m)])
            self.learning_rate = self.learning_schedule(self.learning_rate)
            for i in range((int)(m/self.batch_size)+1):
                begin = i * self.batch_size
                end = (i+1) * self.batch_size if (i+1) * self.batch_size < m else m
                xi = X[begin:end]
                yi = y[begin:end]
                gradient_vector = 2 * xi.T.dot(xi.dot(self.theta) - yi)
                self.theta = self.theta - self.learning_rate * gradient_vector


