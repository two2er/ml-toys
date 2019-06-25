import numpy as np
from supervised.linear_regression import LinearRegression


class RegularizedRegression(LinearRegression):
    """
    Ridge(l2-norm) Lasso(l1-norm) Regularization
    """
    def __init__(self, learning_rate=1e-3, n_epoch=2000, alpha=0.05, r=0.5):
        """
        :param learning_rate:
        :param n_epoch:
        :param alpha: for regularized item
        :param r: only for elastic net
        """
        self.alpha = alpha
        self.r = r
        super(RegularizedRegression, self).__init__(learning_rate, n_epoch)
    
    def _training_method(self, X, y):
        for epoch in range(self.n_epoch):
            gradient_vector = 2 / X.shape[0] * X.T.dot(X.dot(self.theta) - y)\
                              + self.alpha * self._regularized_item()
            self.theta = self.theta - self.learning_rate * gradient_vector

            print('epoch: {}. loss (mse): {}'.format(epoch, self._loss(X.dot(self.theta), y)))
    
    def _regularized_item(self):
        raise NotImplementedError()

    @staticmethod
    def _loss(pred, y):
        # mse
        return np.sum((pred - y) ** 2) / len(y)


class RidgeLinearRegression(RegularizedRegression):
    """
    l2-norm regularization + batch-gradient_descent
    regularized item: |w|^2
    derivative: 2w
    """
    def _regularized_item(self):
        return self.theta


class LassoLinearRegression(RegularizedRegression):
    """
    l1-norm regularization + batch-gradient_descent
    regularized item: ||w||
    derivative: sign(w)
    """
    def _regularized_item(self):
        return np.sign(self.theta)


class ElasticNetRegression(RegularizedRegression):
    """
    combining l1-regularization and l2-regularization
    r * l1 + (1-r) * l2
    """
    def _regularized_item(self):
        return self.r * np.sign(self.theta) + (1-self.r) * self.theta


if __name__ == '__main__':
    import pandas as pd
    dataset = pd.read_csv('../dataset/abalone').values
    X, y = dataset[:, 1:].astype(np.float64), dataset[:, 0]
    y[y != 'M'] = -1.
    y[y == 'M'] = 1.
    y = np.expand_dims(y, axis=1).astype(np.float64)
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=40)
    model = ElasticNetRegression(n_epoch=20, alpha=0.2, r=0.5)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print('mse:', np.sum((pred-test_y)**2) / len(test_y))
