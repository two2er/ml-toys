import numpy as np
import heapq


class KNN:
    """
    k-nearest neighbours
    https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
    """
    def __init__(self, K=1):
        self.K = K

    def predict(self, train_X, train_y, test_X):
        # for every samples in test_X, find K nearest neighbours in train_X
        pred = []
        for test_sample in test_X:
            K_neighbours = []
            for i, train_sample in enumerate(train_X):
                # Euclidean distance between train and test samples
                dist = np.linalg.norm(test_sample - train_sample)
                if len(K_neighbours) < self.K:
                    # push current train sample point into K_neighbours heap
                    # dist is negated to construct a maximum heap on dist
                    heapq.heappush(K_neighbours, (-dist, train_y[i]))
                else:
                    # if dist is smaller than some dist in K_neighbours, replace it
                    if -K_neighbours[0][0] > dist:
                        heapq.heappop(K_neighbours)
                        heapq.heappush(K_neighbours, (-dist, train_y[i]))
            K_labels = [x[1] for x in K_neighbours]
            pred.append(self._val(K_labels))

        return np.array(pred)

    def _val(self, K_neighbours):
        raise NotImplementedError()


class KNNClassifier(KNN):
    """k-nearest neighbours classifier
        simply majority voting
    """
    def _val(self, K_labels):
        values, counts = np.unique(K_labels, return_counts=True)
        idx = np.argmax(counts)
        return values[idx]


class KNNRegressor(KNN):
    """k-nearest neighbours classifier
        mean
    """
    def _val(self, K_labels):
        return np.mean(K_labels)


if __name__ == '__main__':
    import pandas as pd
    dataset = pd.read_csv('../dataset/abalone').values
    dataset = dataset[:int(0.3*len(dataset))]
    X, y = dataset[:, 1:].astype(np.float64), dataset[:, 0]
    y[y != 'M'] = -1.
    y[y == 'M'] = 1.
    y = y.astype(np.float64)
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=40)
    model = KNNClassifier(K=5)
    pred = model.predict(train_X, train_y, test_X)
    print('mse:', np.sum((pred-test_y)**2) / len(test_y))
