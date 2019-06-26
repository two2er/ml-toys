import numpy as np
from supervised.regression_tree import DecisionTreeRegressor


class GradientBoostingMachine:
    """ gbm

    https://en.wikipedia.org/wiki/Gradient_boosting

    F_m(x) = F_{m-1}(x) + learning_rate * lambda

    the base estimator is regression tree, the cost function is MSE.
    so the gradient is (y_pred-y), lambda is -gradient = (y-y_pred)
    """

    def __init__(self, n_estimators=10, learning_rate=1e-2, random_state=-1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        np.random.seed(random_state)
        self.trees = []

    def fit(self, X, y):
        """
        F_1 = tree.fit(X, y).predict(X)

        for m in 2 to n_estimators:
            lambda_i = -gradient = y-F_{m-1}
            h_i = tree.fit(X, lambda_i).predict(X)
            F_m = F_{m-1} + learning_rate*h_i
        """
        # base
        # initialize
        self.F0 = np.mean(y, axis=0)
        Fx = np.full(y.shape[0], self.F0)

        for i in range(self.n_estimators):
            lambda_i = y - Fx
            tree = DecisionTreeRegressor(random_state=np.random.randint(10000))
            tree.fit(X, lambda_i)
            Fx = Fx + self.learning_rate * tree.predict(X)
            self.trees.append(tree)

            print('epoch: {}. loss (mse): {}'.format(i, self._loss(Fx, y)))

    def predict(self, X):
        """ sum of all predictions of each trees """

        prediction = np.full(X.shape[0], self.F0)

        for i in range(self.n_estimators):
            prediction += self.trees[i].predict(X) * self.learning_rate

        return prediction

    @staticmethod
    def _loss(pred, y):
        # mse
        return np.sum((pred - y) ** 2) / len(y)


if __name__ == '__main__':
    import pandas as pd
    dataset = pd.read_csv('../dataset/abalone').values
    X, y = dataset[:, 1:].astype(np.float64), dataset[:, 0]
    y[y != 'M'] = -1.
    y[y == 'M'] = 1.
    y = y.astype(np.float64)
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=40)
    model = GradientBoostingMachine(n_estimators=50, learning_rate=1e-2, random_state=42)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print('mse:', np.sum((pred-test_y)**2) / len(test_y))
