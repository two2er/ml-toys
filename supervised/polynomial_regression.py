import numpy as np
from itertools import combinations_with_replacement
from supervised.linear_regression import BatchGradientRegression


class PolynomialRegression(BatchGradientRegression):
    """
    polynomialization + batch-gradient_descent
    """
    def __init__(self, degree=1, learning_rate=1e-3, n_epoch=2000):
        self.degree = degree
        super(PolynomialRegression, self).__init__(learning_rate, n_epoch)

    def _polynoimalize(self, X):
        # combinations of features (product)
        combinations = [combinations_with_replacement(range(X.shape[1]), d) for d in range(self.degree+1)]
        combinations = [combination for combination_of_a_degree in combinations
                        for combination in combination_of_a_degree]
        X_pol = np.empty((X.shape[0], len(combinations)))
        for i, combination in enumerate(combinations):
            X_pol[:, i:i+1] = np.prod(X[:, combination], axis=1).reshape(-1, 1)

        return X_pol

    def fit(self, X, y):
        """X: n_sample*n_feature, y: n_sample*n_class"""
        super(PolynomialRegression, self).fit(self._polynoimalize(X), y)

    def predict(self, X):
        return super(PolynomialRegression, self).predict(self._polynoimalize(X))


if __name__ == '__main__':
    import pandas as pd
    dataset = pd.read_csv('../dataset/abalone').values
    X, y = dataset[:, 1:].astype(np.float64), dataset[:, 0]
    y[y != 'M'] = -1.
    y[y == 'M'] = 1.
    y = np.expand_dims(y, axis=1).astype(np.float64)
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=40)
    model = PolynomialRegression(n_epoch=20, degree=2)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print('mse:', np.sum((pred-test_y)**2) / len(test_y))
