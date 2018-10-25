# -*- endoding: utf-8 -*-

import numpy as np
import abc
from LinearRegression import LinearRegressor

class RegularizedRegressor(LinearRegressor):
    """
    Ridge(l2-norm) Lasso(l1-norm) Regularization
    """
    def __init__(self, learning_rate = 1e-3, tolerance = 1e-3, alpha = 0.5):
        self.alpha = alpha
        super(RegularizedRegressor, self).__init__(learning_rate, tolerance)
    
    def training_method(self, X, y):
        count = 0
        while True:
            # delete 2 / X.shape[0], otherwise the regularization item would dominate
            gradient_vector =   X.T.dot(X.dot(self.theta) - y)\
                                + self.regularized_item()

            if np.linalg.norm(gradient_vector) < self.tolerance:
                break
            self.theta = self.theta - self.learning_rate * gradient_vector
    
    @abc.abstractmethod
    def regularized_item(self):
        pass

class RidgeLinearRegressor(RegularizedRegressor):
    """
    l2-norm regularization + batch-gradient_descent
    """
    def regularized_item(self):
        return self.alpha * self.theta

class LassoLinearRegressor(RegularizedRegressor):
    """
    l1-norm regularization + batch-gradient_descent
    """
    def regularized_item(self):
        return self.alpha * np.sign(self.theta)

class ElasticNetRegressor(RegularizedRegressor):
    """
    combining l1-regularizaiton and l2-regularization
    r * l1 + (1-r) * l2
    """
    def __init__(self, learning_rate = 1e-3, tolerance = 1e-3, alpha = 0.5, r = 0.5):
        self.r = r
        super(ElasticNetRegressor, self).__init__(learning_rate, tolerance, alpha)

    def regularized_item(self):
        return self.alpha * (self.r * np.sign(self.theta) + (1-self.r) * self.theta)