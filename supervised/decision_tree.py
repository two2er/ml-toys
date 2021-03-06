import numpy as np


class TreeNode:
    """node of decision tree"""
    def __init__(self, X, y, leaf, depth, split_feature=None, split_value=None):
        self.X = X
        self.y = y
        # whether this node is a leaf node or not: true or false
        self.leaf = leaf
        self.depth = depth
        # if the node is an internal node, it must have a split_feature and a split_value,
        # and two children
        self.split_feature = split_feature
        self.split_value = split_value
        self.left_child, self.right_child = None, None
        # if the node is a leaf node, it must have a predict label
        self.leaf_predict = None


class DecisionTree:
    """ base class of DecisionTreeClassifier and DecisionTreeRegressor """
    def __init__(self, min_samples_split=2, min_impurity_decrease=1e-7,
                 max_depth=None, max_features=None, random_state=0):
        """ decision tree classifier based on CART
        min_samples_split: int
            The minimum number of samples needed to make a split when building a tree.
        min_impurity_decrease: float
            A node will be split if this split induces a decrease of the impurity greater
            than or equal to this value.
        max_depth: int
            The maximum depth of a tree. None means inf
        max_features: int
            The maximum number of features considered when splitting a node.
        random_state: int
            Random_state is the seed used by the random number generator.
        """
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_depth = float('inf') if max_depth is None else max_depth
        self.max_features = max_features
        np.random.seed(random_state)
        # the root node of decision tree
        self.root = None

    def fit(self, X, y):
        self.root = self._create_node(X, y, depth=0)
        self._split_node(self.root)

    def _create_node(self, X, y, depth):
        """return a internal/leaf node"""
        # if the number of samples is smaller than min_samples_split, return a leaf node
        # if the depth is larger than max_depth, return a leaf node
        if len(X) < self.min_samples_split or depth >= self.max_depth:
            node = TreeNode(X, y, leaf=True, depth=depth)
            node.leaf_predict = self._leaf_predict(y)
        else:
            node = TreeNode(X, y, leaf=False, depth=depth)

        return node

    def _split_node(self, node):
        """split a node and its descents recursively"""
        if node.leaf:
            return
        # search for a feature for split that would minimize the impurity
        best_feature, best_value, best_impurity_decrease = -1, -1, -float('inf')
        current_impurity = self._impurity(node.y)
        # try max_features features
        feature_seq = np.random.permutation(range(node.X.shape[1]))[:self.max_features]
        for feature in feature_seq:
            feature_values = np.unique(node.X[:, feature])
            # try all feature values
            for feature_value in feature_values:
                # split samples of the node
                left_y = node.y[node.X[:, feature] <= feature_value]
                right_y = node.y[node.X[:, feature] > feature_value]
                if len(left_y) < self.min_samples_split or len(right_y) < self.min_samples_split:
                    continue
                impurity_gain = current_impurity - len(left_y)/len(node.y)*self._impurity(left_y) \
                    - len(right_y)/len(node.y)*self._impurity(right_y)
                if impurity_gain > best_impurity_decrease:
                    best_feature, best_value, best_impurity_decrease = feature, feature_value, impurity_gain

        # if the best impurity gain is smaller than min_impurity_decrease, stop splitting
        if best_impurity_decrease <= self.min_impurity_decrease:
            node.leaf = True
            node.leaf_predict = self._leaf_predict(node.y)
            return

        # split the current node into two children, and split them recursively
        node.left_child = self._create_node(node.X[node.X[:, best_feature] <= best_value],
                                            node.y[node.X[:, best_feature] <= best_value], node.depth+1)
        node.right_child = self._create_node(node.X[node.X[:, best_feature] > best_value],
                                             node.y[node.X[:, best_feature] > best_value], node.depth+1)
        node.split_feature, node.split_value = best_feature, best_value
        self._split_node(node.left_child)
        self._split_node(node.right_child)

    def _impurity(self, y):
        return NotImplementedError()

    def _leaf_predict(self, y):
        return NotImplementedError()

    @staticmethod
    def _loss(pred, y):
        # mse
        return np.sum((pred-y)**2) / len(y)

    def predict(self, X):
        assert self.root, "you must fit the data first before predicting"

        return np.array([self._predict_each(x, self.root) for x in X])

    def _predict_each(self, x, node):
        """return the predict label of one sample"""
        if node.leaf:
            return node.leaf_predict

        if x[node.split_feature] <= node.split_value:
            return self._predict_each(x, node.left_child)
        else:
            return self._predict_each(x, node.right_child)


class DecisionTreeClassifier(DecisionTree):
    """ targets are discrete labels """

    def _impurity(self, y):
        """Gini impurity

        G_i = 1 - sum_{k=1}^n p_{i,k}^2
        where p_{i,k} is the ratio of class k instances among the 
        training instances in the ith node
        """
        unique, counts = np.unique(y, return_counts=True)
        G = 1
        for class_type, count in zip(unique, counts):
            G -= (count/len(y))**2
        return G

    def _leaf_predict(self, y):
        """
        simple majority voting
        """
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]


class DecisionTreeRegressor(DecisionTree):
    """ targets are continuous values """

    def _impurity(self, y):
        """ variance """
        if len(y) == 0:
            return 0
        return np.var(y)

    def _leaf_predict(self, y):
        """ mean value of samples of the leaf node """
        return np.mean(y)


if __name__ == '__main__':
    import pandas as pd
    dataset = pd.read_csv('../dataset/abalone').values
    X, y = dataset[:, 1:].astype(np.float64), dataset[:, 0]
    y[y != 'M'] = -1.
    y[y == 'M'] = 1.
    y = y.astype(np.float64)
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=40)
    model = DecisionTreeRegressor(random_state=41, max_depth=3)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print('loss', model._loss(pred, test_y))
