import numpy as np
from supervised.regression_tree import DecisionTreeRegressor


class RandomForestRegressor:
    """ 
    train n_estimators regression trees. each regression tree would be trained by a subset of the whole
    trainset (sampling with replacement, the size of the subset is equal to the size of the trainset).
    while nodes of decision trees are being split, only max_features features (selected randomly)
    would be considered. while predicting, combine predictions of all predictors by averaging.

    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
    n_samples: float
        The percent of samples to feed a tree.
    max_features: int
        The maximum number of features that the classification trees are allowed to
        use.
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity_decrease: float
        A node will be split if this split induces a decrease of the impurity greater
        than or equal to this value.
    max_depth: int
        The maximum depth of a tree.
    random_state: int
        Random_state is the seed used by the random number generator.
    """

    def __init__(self, n_estimators=100, n_samples=-1, max_features=-1, min_samples_split=2,
                 min_impurity_decrease=1e-7, max_depth=-1, random_state=0):
        self.n_estimators = n_estimators
        self.n_samples = n_samples
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_depth = max_depth
        np.random.seed(random_state)
        self.trees = []

    def fit(self, X, y):
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_features=self.max_features, min_samples_split=self.min_samples_split,
                                         min_impurity_decrease=self.min_impurity_decrease, max_depth=self.max_depth,
                                         random_state=np.random.randint(100000))
            if self.n_samples == -1:
                tree.fit(X, y)
            else:
                tree.fit(*self._sampling_with_replacement(X, y))
            self.trees.append(tree)

            # training log
            print('epoch: {}. loss (mse): {}'.format(i, self._loss(self.predict(X), y)))

    def _sampling_with_replacement(self, X, y):
        # randomly select n_samples samples from X and y with replacement
        random_seq = np.random.randint(X.shape[0], size=int(self.n_samples*X.shape[0]))
        return X[random_seq], y[random_seq]

    @staticmethod
    def _loss(pred, y):
        # mse
        return np.sum((pred - y) ** 2) / len(y)

    def predict(self, X):
        pred = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(pred, axis=0)


if __name__ == '__main__':
    import pandas as pd
    dataset = pd.read_csv('../dataset/abalone').values
    X, y = dataset[:, 1:].astype(np.float64), dataset[:, 0]
    y[y != 'M'] = -1.
    y[y == 'M'] = 1.
    y = y.astype(np.float64)
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=40)
    model = RandomForestRegressor(n_estimators=100, n_samples=0.3,
                                  max_features=4, random_state=42)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print('mse:', np.sum((pred-test_y)**2) / len(test_y))
