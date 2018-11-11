import numpy as np
from itertools import combinations_with_replacement
from LinearRegression import BatchGradientRegressor

class PolynomialRegressor(BatchGradientRegressor):
    """
    polynomialize + batch-gradient_descent
    """
    def __init__(self, degree = 1, learning_rate = 1e-3, n_epoch = 2000):
        self.degree = degree
        super(PolynomialRegressor, self).__init__(learning_rate, n_epoch)

    def _polynoimalize(self, X):
        combinations = [combinations_with_replacement(range(X.shape[1]), d) for d in range(self.degree+1)]
        # for degree 0, the feature value is 1: np.prod(X[:, ()], axis=1) -> [1, .., 1]
        combinations = [combination for combination_of_a_degree in combinations for combination in combination_of_a_degree]
        X_pol = np.empty((X.shape[0], len(combinations)))
        for i, combination in enumerate(combinations):
            X_pol[:, i:i+1] = np.prod(X[:, combination], axis=1).reshape(-1, 1)

        return X_pol

    def fit(self, X, y):
        """X: n_sample*n_feature, y: n_sample*n_class"""
        super(PolynomialRegressor, self).fit(self._polynoimalize(X), y)

    def predict(self, X):
        return super(PolynomialRegressor, self).predict(self._polynoimalize(X))