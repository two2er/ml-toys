import numpy as np
import heapq


class KdNode:
    """
    node of kdtree.
    """
    def __init__(self, depth, splitting_feature, splitting_value, idx, parent):
        """
        :param depth: depth of the node.
        :param splitting_feature: split samples by which feature.
        :param splitting_value: split samples by which feature value.
        :param idx: indices of samples in the dataset.
        :param parent: the parent node if it exists.
        """
        self.depth = depth
        self.splitting_feature = splitting_feature
        self.splitting_value = splitting_value
        self.idx = idx
        self.parent = parent
        # left and right children
        self.left, self.right = None, None
        # 1 if the left child of the node has been visited.
        # 0 if the right child of the node has been visited.
        self.visited = -1


class KdTree:
    """an efficient algorithm of find k-nearest-neighbours
    https://en.wikipedia.org/wiki/K-d_tree

    pseudo-code: (construct)
        input: X, shape is (n_samples, n_features). dimension k
        output: k-d tree

        (1) start: divide all samples in X into two equal-sized collections by the median of the
            first feature. Construct a root whose depth is 1. For samples equal to the median,
            store them at the root. Store samples < median at the left child of the root,
            and those > median at the right child.
        (2) repeat: for nodes of depth j, select the l-th feature as splitting axis. l = j(mod k).
            divide samples in the node by the median of the l-th feature. store samples equal to
            the median at the node, and split other samples into left and right children on whether
            they < median.
        (3) terminate: terminate until no samples in left and right subtrees of the node.

    pseudo-code: (search)
        input: k-d tree, target sample x.
        output: k nearest neighbours of x. (a list 'k-nn')

        (1) top-down: starting from the root. if the feature value of the splitting axis of x is smaller
            than the splitting threshold (the median of 1st feature) of the root, move it to the left
            child. else to the right child. go down recursively until reach a leaf. append samples of
            the leaf to a list 'k-nn'.
        (2) bottom-up: move to the parent of current node. If the max distance from x to samples in
            'k-nn' is larger than the distance from x to the splitting threshold of the parent, search
            for samples in the right subtree which is closer to x than some samples in 'k-nn'. If
            successfully find some, replace those 'furthest' samples in 'k-nn' with closer samples
            if the size of 'k-nn' > k.
        (3) terminate: terminate if reach the root and finish checking its right subtree.
    """
    def __init__(self):
        self.root = None

    def create(self, X, dimensions=None):
        """
        create a kd-tree on data X.
        :param X: shape is (n_samples, n_features).
        :param dimensions: the max number of features chosen for splitting samples. if None, set to
                           be n_features.
        :return: None
        """
        n_samples, n_features = X.shape
        self.X = X
        if not dimensions:
            dimensions = n_features

        self.root = KdNode(depth=0,
                           splitting_feature=0,
                           splitting_value=np.median(X[:, 0]),
                           idx=np.arange(n_samples),
                           parent=None)
        # grow the tree by DFS
        stack = [self.root]
        while stack:
            node = stack.pop()
            # splitting samples in the node into two children
            sample_values = X[node.idx, node.splitting_feature]
            left_idx = node.idx[sample_values < node.splitting_value]
            right_idx = node.idx[sample_values > node.splitting_value]
            node.idx = node.idx[sample_values == node.splitting_value]
            # since left and right subtrees are divided by the median of their parent,
            # the sizes of the two subtrees are expected to be equal
            assert len(left_idx) == len(right_idx),\
                'left and right subtrees should have the same number of samples'
            # append left and right children
            if len(left_idx):
                child_depth = node.depth + 1
                child_feature = (node.depth + 1) % dimensions
                left_value = np.median(X[left_idx, child_feature])
                node.left = KdNode(depth=child_depth, splitting_feature=child_feature,
                                   splitting_value=left_value, idx=left_idx, parent=node)
                right_value = np.median(X[right_idx, child_feature])
                node.right = KdNode(depth=child_depth, splitting_feature=child_feature,
                                    splitting_value=right_value, idx=right_idx, parent=node)
                stack.append(node.left)
                stack.append(node.right)

    def _search(self, x, k=3):
        """
        :param x: the target sample point. shape is (n_features,)
        :param k: the number of nearest neighbours to find.
        :return: a list of k nearest neighbours.
        """
        # top-down
        cur_node = self.root
        # kd-tree is actually a full binary tree
        while cur_node.left:
            if x[cur_node.splitting_feature] <= cur_node.splitting_value:
                cur_node.visited = 1
                cur_node = cur_node.left
            else:
                cur_node.visited = 0
                cur_node = cur_node.right
        # append samples in cur_node into k_nn. k_nn is a max heap
        k_nn = []
        # bottom-top
        while cur_node:
            for idx in cur_node.idx:
                # Euclidean distance
                dist = np.linalg.norm(self.X[idx] - x)
                # negate the dist to construct a max heap
                heapq.heappush(k_nn, (-dist, idx))
            if abs(x[cur_node.splitting_feature] - cur_node.splitting_value) < -k_nn[0][0]:
                # the max distance from x to samples in 'k-nn' > the distance from x to the splitting threshold
                # check samples of another child
                if cur_node.visited:
                    checking_samples = self._samples_of_subtree(cur_node.right, x, k)
                else:
                    checking_samples = self._samples_of_subtree(cur_node.left, x, k)
                k_nn.extend(checking_samples)
                heapq.heapify(k_nn)
                # keep the size of k_nn <= k
                while len(k_nn) > k:
                    heapq.heappop(k_nn)
            cur_node = cur_node.parent
        # sort k_nn
        k_nn.sort(reverse=True)
        dists, idxs = zip(*k_nn)
        return [-d for d in dists], list(idxs)

    def search(self, X, k=3):
        """
        :param X: the target sample points. shape is (n_samples, n_features)
        :param k: the number of nearest neighbours to find.
        :return: lists of k nearest neighbours for each sample point.
        """
        assert self.root, 'must create a tree before search'

        result = [self._search(x, k) for x in X]
        dists, idxs = zip(*result)
        return np.array(dists), np.array(idxs)

    def _samples_of_subtree(self, root, x, k):
        # get k nearest neighbours from the subtree rooted at root
        k_nn = []

        def dfs(node):
            if not node:
                return
            for idx in node.idx:
                dist = np.linalg.norm(x - self.X[idx])
                heapq.heappush(k_nn, (-dist, idx))
            while len(k_nn) > k:
                heapq.heappop(k_nn)
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return k_nn


if __name__ == '__main__':
    from sklearn.neighbors import NearestNeighbors
    n_samples, n_features = 1000, 10
    n_test = 100
    K = 5
    X = np.random.random((n_samples, n_features))
    test_X = np.random.random((n_test, n_features))
    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(test_X)
    tree = KdTree()
    tree.create(X)
    dists, idxs = tree.search(test_X, k=K)
    print(np.all(distances == dists))
    print(np.all(indices == idxs))
