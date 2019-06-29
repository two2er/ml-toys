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

# Mitigate precision differences between 32 bit and 64 bit
cdef DTYPE_t FEATURE_THRESHOLD = 1e-7

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

# DecisionTreeRegressor class

'''
the split criterion is min_Variance, which is equal to mse.

weighted impurity:
N_t / N * (impurity - N_t_R / N_t * right_impurity
                    - N_t_L / N_t * left_impurity)

sample values <= split_point -> left child node
'''

cdef class DecisionTreeRegressor:
    cdef:
        SIZE_t max_depth, min_samples_split, max_features, n_samples, n_features
        DOUBLE_t min_impurity_decrease
        UINT32_t random_state
        Node *root

        DTYPE_t *train
        DOUBLE_t *target
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


    def __cinit__(self, max_depth=-1, min_samples_split=2,
                  min_impurity_decrease=0.0, max_features=-1,
                  random_state=-1):
        """
        max_depth: the maximum depth of the tree
        min_samples_split: the minimum number of samples required to split an internal node
        min_impurity_decrease: a node will be split if this split induces a decrease of the impurity greater than or equal to this value
        max_features: number of features considered when spliting a node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features

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

        self.train, self.target, self.index, self.features, self.Xf = NULL, NULL, NULL, NULL, NULL

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

    def fit(self, train, target):
        """
        the input trainset size would be (sample_num, feature_num) the input target size would be (sample_num,).

        """
        # set dtype, convert to C array (fortran mode)
        train = np.array(train, dtype=np.float32, order='F')
        cdef np.ndarray train_ndarray = train
        self.train = <DTYPE_t*> train_ndarray.data
        self.train_feature_stride = <SIZE_t> train.strides[1] / <SIZE_t> train.itemsize

        target = np.array(target, dtype=np.float64).reshape(-1)
        cdef np.ndarray target_array = target
        self.target = <DOUBLE_t*> target_array.data

        assert self.train != NULL and self.target != NULL, "X and y should not be NULL"

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
            node[0][0] = Node(self._mean_of_target_from_start_to_end(start, end), -1, -1, 1, 0, NULL, NULL, start, end)
            return

        # impurity of this node
        cdef double current_impurity = self._var_of_target_from_start_to_end(start, end)

        cdef:
            SIZE_t i, j, f
            SIZE_t feature

            DOUBLE_t sum_sqr_left, sum_sqr_right, sum_left, sum_right

            SIZE_t current_sample_index    # index of sample considered this iteration
            DTYPE_t current_value          # feature value considered this iteration
            DOUBLE_t current_target        # target of sample considered this iteration
            DTYPE_t next_value             # feature value of index[start+i+1]
                                           # if current_value == next_value, it is useless
                                           # to evaluate impurity with current_sample_index
                                           # as a split point
                                           # (note that we choose (current_value+next_value)/2
                                           #  as the threshold or split value)

            DOUBLE_t left_variance
            DOUBLE_t right_variance
            DOUBLE_t left_propo
            DOUBLE_t right_propo
            DOUBLE_t gain_impurity
            # the current node should be split by the best_feature+best_value, and its best_impurity is the best...
            DOUBLE_t best_impurity = 0.0
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

            # check sort
            # prev = self.train[self.index[start]+self.train_feature_stride*feature]
            # for i in range(start+1, end):
            #     if self.train[self.index[i]+self.train_feature_stride*feature] < prev:
            #         raise ValueError("index sort fail")
            #     prev = self.train[self.index[i]+self.train_feature_stride*feature]
            #
            # prev = self.Xf[start]
            # for i in range(start+1, end):
            #     if self.Xf[i] < prev:
            #         raise ValueError("Xf sort fail")
            #     prev = self.Xf[i]

            # note that self.train[index[start:end], feature] has been sorted.


            # var = 1/n_samples * (\sum_i (y_i ** 2)) - y_bar**2
            #     = 1/n_samples * (\sum_i (y_i ** 2)) - (\sum_i y_i)**2 / n_samples**2
            sum_sqr_left = 0.0
            sum_sqr_right = 0.0
            sum_left = 0.0
            sum_right = 0.0

            for i in range(start, end):
                sum_sqr_right += self.target[self.index[i]] ** 2
                sum_right += self.target[self.index[i]]

            for i in range(n_node_samples-1):
                ## do not consider self.index[n_samples-1+start]
                ## if this sample feature value is a split point, them there would be
                ## n_samples samples to be divided into the left child, and 0 into the right
                ## this should be avoided since we are fitting a binary tree
                ## as well, it is convenience for our calculating

                # the index of sample considered in this iteration
                current_sample_index = self.index[i+start]
                current_value = self.Xf[i+start]
                current_target = self.target[current_sample_index]
                next_value = self.Xf[i+start+1]

                sum_sqr_left += current_target ** 2
                sum_sqr_right -= current_target ** 2
                sum_left += current_target
                sum_right -= current_target

                # we are going to evaluate impurity
                # we think that current_value != next_value
                if current_value + FEATURE_THRESHOLD < next_value:
                    left_variance = (sum_sqr_left - sum_left ** 2 / (i+1)) / (i+1)
                    right_variance = (sum_sqr_right - sum_right ** 2 / (n_node_samples-i-1)) / (n_node_samples-i-1)
                    left_propo = <double>(i+1) / n_node_samples
                    right_propo = 1 - left_propo

                    gain_impurity = current_impurity - (left_propo*left_variance + right_propo*right_variance)

                    if gain_impurity > best_impurity:
                        best_feature, best_value, best_impurity = feature, (current_value+next_value)/2, gain_impurity

        # if the best_impurity is smaller than min_impurity_decrease, it is a leaf
        ## note that order: best_impurity/self.n_samples*n_node_samples
        ## if the order is different, it might be 0 because self.n_samples and n_node_samples are int
        ## i.e. n_node_samples/self.n_samples*best_impurity
        if best_impurity/self.n_samples*n_node_samples < self.min_impurity_decrease + FEATURE_THRESHOLD:
            node[0][0] = Node(self._mean_of_target_from_start_to_end(start, end), -1, -1, 1, 0, NULL, NULL, start, end)
            return

        node[0][0] = Node(-1, best_feature, best_value, 0, 0, NULL, NULL, start, end)
        return

    cdef void _split(self, Node **node):
        # if it is a leaf node, end splitting
        if node[0].leaf:
            return
        # if the depth >= max_depth, stop
        if node[0].depth >= self.max_depth:
            node[0].leaf = 1
            node[0].val = self._mean_of_target_from_start_to_end(node[0].start, node[0].end)
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

        assert node[0].start != pos and pos != node[0].end, "fail to choose a valid split point"

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

    cdef DOUBLE_t _mean_of_target_from_start_to_end(self, SIZE_t start, SIZE_t end):
        cdef DOUBLE_t _mean = 0.0
        cdef SIZE_t i
        for i in range(start, end):
            _mean += self.target[self.index[i]]

        return _mean / (end-start)

    cdef DOUBLE_t _var_of_target_from_start_to_end(self, SIZE_t start, SIZE_t end):
        cdef DOUBLE_t _var = 0.0
        cdef DOUBLE_t _mean = self._mean_of_target_from_start_to_end(start, end)
        cdef SIZE_t i
        for i in range(start, end):
            _var += (self.target[self.index[i]] - _mean) ** 2

        return _var / (end-start)

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
