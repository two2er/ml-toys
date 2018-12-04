import random
import numpy as np

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
    r'''
        the change of objective after adding a split:
        Gain = G_L^2/(H_L+\lambda) + G_R^2/(H_R+\lambda) - (G_L+G_R)^2/(H_L+H_R+\lambda) - \gamma

        where: G_L = \sum_{ith sample on the left child}g_i
               H_L = \sum_{ith sample on the left child}h_i
               \lambda: the weight to punish number of leaves
               \gamma: the weight to punish sum of leaf scores
    '''
    def __init__(self, max_depth=None,
                 min_samples_split=2,
                 min_object_decrease=0.0,
                 random_state=None,
                 _lambda=1, _gamma=0.0, shrinkage=1.):
        '''
        max_depth: the maximum depth of the tree
        min_samples_split: the minimum number of samples required to split an internal node
        min_object_decrease: a node will be split if this split induces a decrease of the objective greater than or equal to this value
        '''
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_object_decrease = min_object_decrease
        
        if random_state != None:
            random.seed(random_state)
        
        # self.max_depth == None means no fixed max_depth
        if self.max_depth == None:
            self.max_depth = np.inf
        
        self._lambda = _lambda
        self._gamma = _gamma
        self.shrinkage = shrinkage

        self.root = None
        
        
    def fit(self, train, G, H):
        self.train = train
        self.G = G
        self.H = H

        self.N, self.feature_num = self.train.shape
        
        # index: the index of all trainsets in a node
        index = np.arange(self.N)
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
        # decide whether it is splitable. If it is, it is an internal node. else it is a leaf node
        if len(index) < self.min_samples_split:
            return Node(val=self._weight(index)*self.shrinkage, leaf=True)
        
        current_object = self._object(index)

        # the node would be split by best_feature+threshold
        best_feature, threshold, best_object_gain = None, None, -np.inf
        
        for feature in range(self.feature_num):
            
            # sort index based on feature values
            index = index[np.argsort(self.train[index, feature])]
            
            # note that self.train[index, feature] has been sorted. if we meet a new value, we split values before
            # it and values after it into two children
            current_value = None
            
            for i in range(len(index)):
                # we meet a new value
                if self.train[index[i], feature] != current_value:
                    # we use it as a new split value
                    left_object = self._object(index[:i])
                    right_object = self._object(index[i:])
                    gain_object = left_object + right_object - current_object - self._gamma
                    if gain_object > best_object_gain:
                        best_feature, threshold, best_object_gain = feature, self.train[index[i], feature], gain_object
                        best_index = index

                    current_value = self.train[index[i], feature]
                
        # if the best_object_gain is smaller than min_object_decrease, it is a leaf
        if len(index)/self.N * best_object_gain <= self.min_object_decrease:
            return Node(val=self._weight(index)*self.shrinkage, leaf=True)
        
        # return an internal node
        return Node(index=best_index, split_feature=best_feature, split_value=threshold, leaf=False)
          
        
    def split(self, node):
        # if it is a leaf node, end spliting
        if node.leaf:
            return
        # if the depth >= max_depth, stop
        if node.depth >= self.max_depth:
            node.leaf = True
            node.val = self._weight(node.index)*self.shrinkage
            return
        
        # split the index of node into two groups: left child and right child
        # as node.index is sorted, we can just count how many indices smaller than split_value
        count = np.where(self.train[node.index, node.split_feature] == node.split_value)[0][0]
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
    def __init__(self, n_estimators=10, max_depth=None,
                 min_samples_split=2,
                 min_object_decrease=0,
                 random_state=None,
                 _lambda=1, _gamma=0.0,
                 learning_rate=0.1):
        self.n_estimators = n_estimators

        self.min_samples_split = min_samples_split
        self.min_object_decrease = min_object_decrease

        self.random_state = random_state
        self.max_depth = max_depth
        
        self._lambda = _lambda
        self._gamma = _gamma
        self.learning_rate = learning_rate

        self.base_estimator = GBDT_tree

    def fit(self, train, label):
        train = np.array(train)
        label = np.array(label)
        m, n = train.shape
        # shrinking learning rate
        shrinkage = self.learning_rate

        # a list of base estimators (length = n_estimators)
        self.trees = []
        
        # \hat{y}_{t-1}. initialized with mean
        self.mean_of_train = np.mean(label)
        hat_y = np.zeros(m)
        
        # calculate G and H for each base estimator
        # since h_i is always 2, we can fix H to a constant array
        H = np.full(m, 2)
        for i in range(self.n_estimators):
            if i == 0:
                G = np.random.uniform(size=m)
            else:
                G = self._getG(hat_y, label)

            if i < 0:
                print(i, G)

            tree = self.base_estimator(self.max_depth, self.min_samples_split,
                                       self.min_object_decrease, self.random_state,
                                       self._lambda, self._gamma, shrinkage)

            tree.fit(train, G, H)
            self.trees.append(tree)

            # update hat_y
            pred = tree.predict(train)
            hat_y += pred

            shrinkage = self.learning_rate

            if i < 0:
                print(i, pred)

            if i < 0:
                print(hat_y)

            if i >= 0:
                print(self._mse(hat_y, label))
            
    def _mse(self, hat_y, label):
        return np.linalg.norm(hat_y-label)
        # return np.mean(np.square(hat_y-label))

    def _getG(self, hat_y, label):
        # if len(self.trees) == 0:
        #     return np.random.uniform(size=len(hat_y))
        rtn = np.empty(len(hat_y), dtype=np.float)
        for i in range(len(hat_y)):
            rtn[i] = 2 * (hat_y[i] - label[i])
        return rtn

    def predict(self, test, test_y):
        rtn = np.zeros(len(test))
        print(self._mse(rtn, test_y))
        for i in range(self.n_estimators):
            pred = self.trees[i].predict(test)
            rtn += pred
            # shrinkage *= self.learning_rate
            print(self._mse(rtn, test_y))

        return rtn