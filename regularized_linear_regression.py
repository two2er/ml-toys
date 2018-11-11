import numpy as np
from LinearRegression import LinearRegressor

class RegularizedRegressor(LinearRegressor):
    """
    Ridge(l2-norm) Lasso(l1-norm) Regularization
    """
    def __init__(self, learning_rate = 1e-3, n_epoch=2000, alpha = 0.05):
        self.alpha = alpha
        super(RegularizedRegressor, self).__init__(learning_rate, n_epoch)
    
    def _training_method(self, X, y):
        for epoch in range(self.n_epoch):
            gradient_vector =   2 / X.shape[0] * X.T.dot(X.dot(self.theta) - y)\
                                    + self._regularized_item()
            self.theta = self.theta - self.learning_rate * gradient_vector
    
    def _regularized_item(self):
        raise NotImplementedError()

class RidgeLinearRegressor(RegularizedRegressor):
    """
    l2-norm regularization + batch-gradient_descent
    """
    def _regularized_item(self):
        return self.alpha * self.theta

class LassoLinearRegressor(RegularizedRegressor):
    """
    l1-norm regularization + batch-gradient_descent
    """
    def _regularized_item(self):
        return self.alpha * np.sign(self.theta)

class ElasticNetRegressor(RegularizedRegressor):
    """
    combining l1-regularizaiton and l2-regularization
    r * l1 + (1-r) * l2
    """
    def __init__(self, learning_rate = 1e-3, n_epoch = 2000, alpha = 0.5, r = 0.5):
        self.r = r
        super(ElasticNetRegressor, self).__init__(learning_rate, n_epoch, alpha)

    def _regularized_item(self):
        return self.alpha * (self.r * np.sign(self.theta) + (1-self.r) * self.theta)