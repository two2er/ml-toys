""" random forest. based on decision tree classifier + majority voting """

import numpy as np
from decision_tree import DecisionTreeClassifier

class RandomForestClassifier:
    """ 
    train n_estimators decision trees. each decision tree would be trained by a subset of the whole
    trainset (sampling with replacement, the size of the subset is equal to the size of the trainset).
    while nodes of decision trees are being split, only max_features features (selected randomly)
    would be considered. while predicting, combine predictions of all predictors by majority voting.

    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
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

    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2,
                 min_impurity_decrease=1e-7, max_depth=None, random_state=0):

        self.trees = [DecisionTreeClassifier(max_features=max_features, min_samples_split=min_samples_split,
                                   min_impurity_decrease=min_impurity_decrease, max_depth=max_depth,
                                   random_state=random_state) for _ in range(n_estimators)]
        self.n_estimators = n_estimators

    def fit(self, X, y):
        for i in range(self.n_estimators):
            self.trees[i].fit(*self._sampling_with_replacement(X, y))

    def _sampling_with_replacement(self, X, y):
        # randomly select n_samples samples from X and y with replacement
        random_seq = np.random.randint(X.shape[0], size=X.shape[0])
        return X[random_seq], y[random_seq]

    def predict(self, X):
        preds = [tree.predict(X) for tree in self.trees]
        pred = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            pred_for_xi = [preds[j][i] for j in range(self.n_estimators)]
            unique, counts = np.unique(pred_for_xi, return_counts=True)
            pred[i] = unique[np.argmax(counts)]
        return pred
