import random
import numpy as np
from supervised.gbdt_tree import GBDTRegressor

class Node:
    def __init__(self, val=None, split_feature=None, split_value=None, index=None, leaf=False, depth=0):
        self.leaf = leaf
        self.val = val
        self.split_feature = split_feature
        self.split_value = split_value
        self.index = index
        self.left = None
        self.right = None
        self.depth = depth
        

class GBDT_tree:
    r"""
        the change of objective function value after adding a split:
        Gain = G_L^2/(H_L+\lambda) + G_R^2/(H_R+\lambda) - (G_L+G_R)^2/(H_L+H_R+\lambda) - \gamma

        where: G_L = \sum_{ith sample on the left child}g_i
               H_L = \sum_{ith sample on the left child}h_i
               \lambda: the weight to punish number of leaves
               \gamma: the weight to punish sum of leaf scores
    """
    def __init__(self, max_depth=-1,
                 min_samples_split=2,
                 min_score_gain=0.0,
                 random_state=-1,
                 _lambda=1, _gamma=0.0):
        """
        max_depth: the maximum depth of the tree
        min_samples_split: the minimum number of samples required to split an internal node
        min_score_gain: a node will be split if this split induces a gain of the score greater than or equal to this value
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_score_gain = min_score_gain
        
        if random_state != -1:
            random.seed(random_state)
        
        # self.max_depth == None means no fixed max_depth
        if self.max_depth == -1:
            self.max_depth = np.inf
        
        self._lambda = _lambda
        self._gamma = _gamma

        self.root = None

    def fit(self, train, G, H):
        self.train = train
        self.G = G
        self.H = H

        self.n_samples, self.n_features = self.train.shape
        
        # index: the index of all trainsets in a node
        index = np.arange(self.n_samples)
        self.root = self.create_node(index)
        # split
        self.split(self.root)
        
    def _weight(self, index):
        # return weight of a node
        return -np.sum(self.G[index]) / (np.sum(self.H[index])+self._lambda)

    def _object(self, index):
        # return the objective function of splitting
        return np.sum(self.G[index])**2 / (np.sum(self.H[index])+self._lambda)

    def create_node(self, index):
        # process the given index, and given a node
        # decide whether it is splittable. If it is, it is an internal node. else it is a leaf node
        if len(index) < self.min_samples_split:
            return Node(val=self._weight(index), leaf=True)
        
        current_object = self._object(index)

        # the node would be split by best_feature+threshold
        best_feature, threshold, best_object_gain = None, None, -np.inf
        
        for feature in range(self.n_features):
            
            # sort index based on feature values
            index = index[np.argsort(self.train[index, feature])]
            
            for i in range(len(index)-1):
                # we meet a new value
                if self.train[index[i], feature] != self.train[index[i+1], feature]:
                    # we use it as a new split value
                    left_object = self._object(index[:i])
                    right_object = self._object(index[i:])
                    gain_object = left_object + right_object - current_object - self._gamma
                    if gain_object > best_object_gain:
                        best_feature, threshold, best_object_gain = feature, self.train[index[i], feature], gain_object
                        best_index = index
                
        # if the best_object_gain is smaller than min_score_gain, it is a leaf
        if len(index)/self.n_samples * best_object_gain <= self.min_score_gain:
            return Node(val=self._weight(index), leaf=True)
        
        # return an internal node
        return Node(index=best_index, split_feature=best_feature, split_value=threshold, leaf=False)
        
    def split(self, node):
        # if it is a leaf node, end splitting
        if node.leaf:
            return
        # if the depth >= max_depth, stop
        if node.depth >= self.max_depth:
            node.leaf = True
            node.val = self._weight(node.index)
            return
        
        # split the index of node into two groups: left child and right child
        # as node.index is sorted, we can just count how many indices smaller than split_value
        count = 0
        while count < len(node.index):
            if self.train[node.index[count], node.split_feature] > node.split_value:
                break
            count += 1
        left = node.index[:count]
        right = node.index[count:]
            
        # recursively split
        node.left = self.create_node(left)
        node.right = self.create_node(right)
        node.left.depth, node.right.depth = node.depth + 1, node.depth + 1
        # delete node.index because it is useless now
        del node.index
        self.split(node.left)
        self.split(node.right)
        
    def predict(self, test):
        return np.array([self.predict_each(x, self.root) for x in test])
    
    def predict_each(self, x, node):
        if node.leaf:
            return node.val
        if x[node.split_feature] < node.split_value:
            return self.predict_each(x, node.left)
        else:
            return self.predict_each(x, node.right)


class GBDT:
    r"""https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf

    Goal:
        Obj^t = \sum_{i=1}^n L(y_i, \hat{y}^{t-1}_i+f_t(x_i)) + \omega(f_t) + constant

    for each sample:
        g_i = \delta_{\hat{y}^{t-1}_i}L(y_i, \hat{y}^{t-1}_i)
        h_i = \delta_{\hat{y}^{t-1}_i}g_i
    
    if the loss function L is MSE:
        g_i = 2(\hat{y}^{t-1}_i-y_i)
        h_i = 2
    """
    def __init__(self, n_estimators=10, max_depth=-1,
                 min_samples_split=2,
                 min_score_gain=0,
                 max_features=-1,
                 sub_samples=1.,
                 random_state=-1,
                 _lambda=1., _gamma=0.0,
                 learning_rate=1e-3):
        self.n_estimators = n_estimators

        self.min_samples_split = min_samples_split
        self.min_score_gain = min_score_gain

        self.random_state = random_state
        self.max_depth = max_depth
        self.max_features = max_features    # feature bagging
        self.sub_samples = sub_samples      # bagging
        
        self._lambda = _lambda
        self._gamma = _gamma
        self.learning_rate = learning_rate

        # self.base_estimator = GBDT_tree
        self.base_estimator = GBDTRegressor

    def fit(self, train, label):
        train = np.array(train)
        label = np.array(label)
        m, n = train.shape

        # a list of base estimators (length = n_estimators)
        self.trees = []
        
        # \hat{y}_{t-1}. initialized with mean
        self.mean_of_train = np.mean(label)
        hat_y = np.full(m, self.mean_of_train)
        
        # calculate G and H for each base estimator
        # since h_i is always 2, we can fix H to a constant array
        H = np.full(m, 2)
        for i in range(self.n_estimators):
            G = self._getG(hat_y, label)

            tree = self.base_estimator(self.max_depth, self.min_samples_split,
                                       self.min_score_gain, self.max_features,
                                       self._lambda, self._gamma, self.random_state)
            # tree = GBDT_tree(self.max_depth, self.min_samples_split,
            #                            self.min_score_gain, self.random_state,
            #                            self._lambda, self._gamma)
            random_seq = np.random.permutation(m)[:int(m*self.sub_samples)]
            tree.fit(train[random_seq], G[random_seq], H[random_seq])
            self.trees.append(tree)

            # update hat_y
            pred = tree.predict(train)
            hat_y += self.learning_rate * pred

            # training log
            print('epoch: {}. loss (mse): {}'.format(i, self._loss(hat_y, label)))

    def _getG(self, hat_y, label):
        # mse derivative
        return 2 * (hat_y - label)

    def predict(self, test):
        rtn = np.full(len(test), self.mean_of_train)
        for tree in self.trees:
            rtn += self.learning_rate * tree.predict(test)

        return rtn

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
    model = GBDT(n_estimators=200, learning_rate=5e-3, random_state=44, _lambda=5e-1, _gamma=5e-1,
                 max_features=4, sub_samples=0.3)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print('mse:', np.sum((pred-test_y)**2) / len(test_y))
