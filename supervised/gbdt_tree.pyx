# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as np
np.import_array()

from libc.stdlib cimport realloc, free

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# dytpe

DEF INF = 0x7FFFFFFF
ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, parameter
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# get random int

cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>INF + 1)

cdef inline SIZE_t rand_int(SIZE_t low, SIZE_t high, UINT32_t* random_state) nogil:
    """Generate a random integer in [low; end)."""
    return low + our_rand_r(random_state) % (high - low)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# struct Node
cdef struct Node:
    # value if it is a leaf
    DOUBLE_t val
    SIZE_t split_feature
    DTYPE_t split_value
    # whether it is a leaf
    SIZE_t leaf
    SIZE_t depth
    # children
    Node *left
    Node *right
    SIZE_t start
    SIZE_t end                  # contains samples index[start:end]
                                # a leaf has no need to know its start and end

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# GBDT tree class (base estimator)

"""
https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf

Gain = G_L^2/(H_L+\lambda) + G_R^2/(H_R+\lambda) - (G_L+G_R)^2/(H_L+H_R+\lambda) - \gamma

        where: G_L = \sum_{ith sample on the left child}g_i
               H_L = \sum_{ith sample on the left child}h_i
               \lambda: the weight to punish number of leaves
               \gamma: the weight to punish sum of leaf scores

weight = -np.sum(self.G[index]) / (np.sum(self.H[index])+self._lambda)
"""

cdef class GBDTRegressor:
    cdef:
        SIZE_t max_depth, min_samples_split, max_features, n_samples, n_features
        DOUBLE_t min_score_gain
        UINT32_t random_state
        Node *root

        DTYPE_t *train
        DTYPE_t *G, *H
        # fortran mode: self.train[i+train_feature_stride*j] == train[i, j]
        SIZE_t train_feature_stride
        # index of all samples
        SIZE_t *index
        # for building tree
        SIZE_t *features    # features for sampling
        DTYPE_t *Xf         # feature values
        # for predicting
        DTYPE_t *test
        SIZE_t test_sample_stride
        DOUBLE_t *predict_result
        SIZE_t test_n_samples

        # parameters of GBDT
        DTYPE_t _lambda, _gamma


    def __cinit__(self, max_depth=-1, min_samples_split=2,
                  min_score_gain=0.0, max_features=-1,
                  _lambda=0.1, _gamma=1., random_state=-1):
        """
        max_depth: the maximum depth of the tree
        min_samples_split: the minimum number of samples required to split an internal node
        min_score_gain: a node will be split if this split induces a gain of the score greater than or equal to this value
        max_features: number of features considered when splitting a node
        _lambda: weight of L2 norm regularization on leaf scores
        _gamma: regularization on number of leaves
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_score_gain = min_score_gain
        self.max_features = max_features
        self._lambda = _lambda
        self._gamma = _gamma

        # set random state
        if random_state != -1:
            self.random_state = <UINT32_t>random_state
        else:
            self.random_state = <UINT32_t>np.random.randint(1, 10000)

        self.root = NULL

        # self.max_depth == -1 means no fixed max_depth
        if self.max_depth == -1:
            self.max_depth = INF

        self.n_samples = -1
        self.n_features = -1

        self.train, self.G, self.H, self.index, self.features, self.Xf = NULL, NULL, NULL, NULL, NULL, NULL

        self.test, self.predict_result = NULL, NULL

    def __dealloc__(self):
        free(self.index)
        free(self.features)
        free(self.Xf)

        self.dealloc_nodes(self.root)

    cdef void dealloc_nodes(self, Node *node):
        if node == NULL:
            return
        self.dealloc_nodes(node.left)
        self.dealloc_nodes(node.right)
        free(node)

    def fit(self, train, G, H):
        """
        the input feature size would be (sample_num, feature_num)
        Gradient, Hessian: (sample_num, )
        """
        # set DType, convert to C array (fortran mode)
        train = np.array(train, dtype=np.float32, order='F')
        cdef np.ndarray train_ndarray = train
        self.train = <DTYPE_t*> train_ndarray.data
        self.train_feature_stride = <SIZE_t> train.strides[1] / <SIZE_t> train.itemsize

        G = np.array(G, dtype=np.float32).reshape(-1)
        cdef np.ndarray G_array = G
        self.G = <DTYPE_t*> G_array.data

        H = np.array(H, dtype=np.float32).reshape(-1)
        cdef np.ndarray H_array = H
        self.H = <DTYPE_t*> H_array.data

        assert self.train != NULL and self.G != NULL and self.H != NULL, "X, G, H should not be NULL"

        self.n_samples = train.shape[0]
        self.n_features = train.shape[1]

        # self.max_features == -1 means use all features when splitting a node
        if self.max_features == -1:
            self.max_features = self.n_features

        # index: the index of all samples in a node
        self.index = <SIZE_t*>realloc(self.index, self.n_samples*sizeof(SIZE_t))
        assert self.index != NULL, "fail to allocate memory to index array"

        cdef SIZE_t i
        for i in range(self.n_samples):
            self.index[i] = i

        # the first node (root) would be created from samples[start, end]
        cdef SIZE_t start = 0, end = self.n_samples

        # arrays for building tree
        self.features = <SIZE_t*>realloc(self.features, self.n_features*sizeof(SIZE_t))
        assert self.features != NULL, "fail to allocate memory to features array"
        for i in range(self.n_features):
            self.features[i] = i

        self.Xf = <DTYPE_t*>realloc(self.Xf, self.n_samples*sizeof(DTYPE_t))
        assert self.Xf != NULL, "fail to allocate memory to Xf array"

        self._create_node(&self.root, start, end)
        # split
        self._split(&self.root)


    cdef void _create_node(self, Node **node, SIZE_t start, SIZE_t end):

        # print('creating node:', start, end)

        assert start != end, "a node must have some samples"

        # allocate memory
        node[0] = <Node*>realloc(node[0], sizeof(Node))
        assert node != NULL, "fail to allocate memory for node"

        # num of samples in this node
        cdef SIZE_t n_node_samples = end - start

        # decide whether it is splittable. If it is, it is an internal node. else it is a leaf node
        if n_node_samples < self.min_samples_split:
            node[0][0] = Node(self._weight(start, end), -1, -1, 1, 0, NULL, NULL, start, end)
            return

        # score of this node
        cdef double current_score = self._score(start, end)

        cdef:
            SIZE_t i, j, f
            SIZE_t feature

            DOUBLE_t sum_left_G, sum_right_G, sum_left_H, sum_right_H

            SIZE_t current_sample_index    # index of sample considered this iteration
            DTYPE_t current_value          # feature value considered this iteration
            DOUBLE_t current_G, current_H  # G, H of sample considered this iteration
            DTYPE_t next_value             # feature value of index[start+i+1]
                                           # if current_value == next_value, it is useless
                                           # to evaluate score with current_sample_index
                                           # as a split point

            DOUBLE_t left_score
            DOUBLE_t right_score
            DTYPE_t gain_score
            # the current node should be split by the best_feature+best_value, and its best_score is the best...
            DTYPE_t best_score = 0
            DTYPE_t best_value
            SIZE_t best_feature

        for j in range(self.max_features):
            # randomly select a feature from self.features[j:self.n_features]
            f = rand_int(j, self.n_features, &self.random_state)
            feature = self.features[f]      # current feature
            # switch self.features[f] and self.features[j], avoid to be sampled again
            self.features[j], self.features[f] = self.features[f], self.features[j]
            # copy feature values to self.Xf
            for i in range(start, end):
                self.Xf[i] = self.train[self.index[i]+self.train_feature_stride*feature]

            self._qsort_index_from_start_to_end_by_feature_value(start, end-1)

            # note that self.train[index[start:end], feature] has been sorted.

            # score = (G**2) / (H + lambda)
            sum_left_G = 0.0
            sum_right_G = 0.0
            sum_left_H = 0.0
            sum_right_H = 0.0

            for i in range(start, end):
                sum_right_G += self.G[self.index[i]]
                sum_right_H += self.H[self.index[i]]

            for i in range(n_node_samples-1):
                ## do not consider self.index[n_samples-1+start]
                ## if this sample feature value is a split point, them there would be
                ## n_samples samples to be divided into the left child, and 0 into the right
                ## this should be avoided since we are fitting a binary tree
                ## as well, it is convenience for our calculating

                # the index of sample considered in this iteration
                current_sample_index = self.index[i+start]
                current_value = self.Xf[i+start]
                current_G = self.G[current_sample_index]
                current_H = self.H[current_sample_index]
                next_value = self.Xf[i+start+1]

                sum_left_G += current_G
                sum_left_H += current_H
                sum_right_G -= current_G
                sum_right_H -= current_H

                # we are going to evaluate score
                if current_value != next_value:

                    left_score = sum_left_G ** 2 / (sum_left_H + self._lambda)
                    right_score = sum_right_G ** 2 / (sum_right_H + self._lambda)
                    gain_score = left_score + right_score - current_score - self._gamma

                    if gain_score > best_score:
                        best_feature, best_value, best_score = feature, current_value, gain_score

        if best_score/self.n_samples*n_node_samples <= self.min_score_gain:
            node[0][0] = Node(self._weight(start, end), -1, -1, 1, 0, NULL, NULL, start, end)
            return

        node[0][0] = Node(-1, best_feature, best_value, 0, 0, NULL, NULL, start, end)
        return

    cdef DTYPE_t _weight(self, SIZE_t start, SIZE_t end):
        return -self._sum(self.G, start, end) / (self._sum(self.H, start, end) + self._lambda)

    cdef DTYPE_t _sum(self, DTYPE_t *pointer, SIZE_t start, SIZE_t end):
        cdef SIZE_t i = start
        cdef DTYPE_t rtn = 0
        while i < end:
            rtn += pointer[self.index[i]]
            i += 1
        return rtn

    cdef DTYPE_t _score(self, SIZE_t start, SIZE_t end):
        return self._sum(self.G, start, end) ** 2 / (self._sum(self.H, start, end) + self._lambda)

    cdef void _split(self, Node **node):
        # if it is a leaf node, end splitting
        if node[0].leaf:
            return
        # if the depth >= max_depth, stop
        if node[0].depth >= self.max_depth:
            node[0].leaf = 1
            node[0].val = self._weight(node[0].start, node[0].end)
            return

        # partition self.index[node[0].start, node[0].end] into two parts: self.index[node[0].start, pos]
        # and self.index[pos, node[0].end]
        # feature values (node[0].split_feature) of samples of self.index[node[0].start, pos]
        # are <= node[0].split_value
        # feature values (node[0].split_feature) of samples of self.index[pos, node[0].end]
        # are > node[0].split_value

        assert node[0].start != node[0].end, "splitting a empty node"

        cdef SIZE_t pos = node[0].start, high = node[0].end
        cdef DTYPE_t pos_feature_value
        while pos < high:
            # index[start, pos]: <= split_value; index[high, end]: > split_value
            pos_feature_value = self.train[self.index[pos]+self.train_feature_stride*node[0].split_feature]
            if pos_feature_value <= node[0].split_value:
                pos += 1
            else:
                high -= 1
                self.index[pos], self.index[high] = self.index[high], self.index[pos]

        self._create_node(&node[0].left, node[0].start, pos)
        self._create_node(&node[0].right, pos, node[0].end)
        node[0].left.depth = node[0].depth + 1
        node[0].right.depth = node[0].depth + 1

        self._split(&node[0].left)
        self._split(&node[0].right)

    cpdef predict(self, object test):
        assert self.root != NULL, "train before predict"

        test = np.array(test, dtype=np.float32)
        self.test_n_samples = test.shape[0]
        assert test.shape[1] == self.n_features, "testset must have the same feature num as trainset"

        cdef np.ndarray test_ndarray = test
        self.test = <DTYPE_t*> test_ndarray.data
        self.test_sample_stride = <SIZE_t> test.strides[0] / <SIZE_t> test.itemsize
        self.predict_result = <DOUBLE_t*>realloc(self.predict_result, test.shape[0]*sizeof(DOUBLE_t))
        assert self.predict_result != NULL, "fail to allocate self.predict_result"

        cdef SIZE_t i
        cdef Node *node
        for i in range(self.test_n_samples):
            node = self.root
            while not node.leaf:
                if self.test[i*self.test_sample_stride+node.split_feature] <= node.split_value:
                    node = node.left
                else:
                    node = node.right
            self.predict_result[i] = node.val

        return self._predict_ptr_to_ndarray()

    cdef void _qsort_index_from_start_to_end_by_feature_value(self, SIZE_t low, SIZE_t high):
        # high = end - 1
        cdef SIZE_t i = low, j = high
        cdef DTYPE_t mid = self.Xf[(low+high)>>1]

        while i <= j:
            while self.Xf[i] < mid:
                i += 1
            while self.Xf[j] > mid:
                j -= 1

            if i <= j:
                self.Xf[i], self.Xf[j] = self.Xf[j], self.Xf[i]
                self.index[i], self.index[j] = self.index[j], self.index[i]
                i += 1
                j -= 1

        if i < high:
            self._qsort_index_from_start_to_end_by_feature_value(i, high)
        if low < j:
            self._qsort_index_from_start_to_end_by_feature_value(low, j)

    cdef inline np.ndarray _predict_ptr_to_ndarray(self):
        """Return copied data as 1D numpy array of self.predict pointer.

        PyArray_SimpleNewFromData: return an array wrapper around data pointed
        to by the pointer 'data'. its shape is 'shape', length is '1', data type
        is 'DOUBLE_t'.
        return a copy of the array.
        """

        cdef SIZE_t shape[1]
        shape[0] = <SIZE_t> self.test_n_samples
        return np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, self.predict_result).copy()
