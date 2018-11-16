import numpy as np

from decision_tree import DecisionTreeRegressor
# from sklearn.tree import DecisionTreeRegressor

class GradientBoostingMachine:
    """ gbm

    https://en.wikipedia.org/wiki/Gradient_boosting

    F_m(x) = F_{m-1}(x) + learning_rate * lambda

    the base estimator is regression tree, the cost function is MSE.
    so the gradient is (y_pred-y), lambda is -gradient = (y-y_pred)
    """

    def __init__(self, n_estimators=10, learning_rate=1e-2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
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
        tree = DecisionTreeRegressor()
        tree.fit(X, y)
        Fx = tree.predict(X)
        self.trees.append(tree)

        for i in range(1, self.n_estimators):
            lambda_i = y - Fx
            tree = DecisionTreeRegressor()
            tree.fit(X, lambda_i)
            Fx = Fx + self.learning_rate*tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        """ sum of all predictions of each trees """

        prediction = self.trees[0].predict(X)

        for i in range(1, self.n_estimators):
            prediction += self.trees[i].predict(X) * self.learning_rate

        return prediction


