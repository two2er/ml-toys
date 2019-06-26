import numpy as np


class Perceptron:
    """binary Perceptron model.

    loss function: -\sum_{x_i misclassified} y_i*(w*x_i+b). sum of distance from the
                   misclassified samples to the hyperplane
    """
    def __init__(self, learning_rate=1e-3, n_epoch=-1):
        """
        :param learning_rate:
        :param n_epoch: if n_epoch == -1, the training process would not be terminated
                        until there is no misclassified sample
        """
        self.learning_rate = learning_rate
        if n_epoch == -1:
            self.n_epoch = 0x7fffffff
        else:
            self.n_epoch = n_epoch
        self.theta = None

    def fit(self, X, y):
        """
        X: n_sample*n_feature, y: n_sample*1
        """
        # add x_0 = 1 for samples
        X_b = np.insert(X, 0, 1, axis=1)
        del X
        # theta: randomly initialized: [-1/sqrt(n), 1/sqrt(n)]
        limit = np.sqrt(X_b.shape[1])
        self.theta = np.random.uniform(-1 / limit, 1 / limit, (X_b.shape[1], 1))

        for epoch in range(self.n_epoch):
            # collect misclassified samples in each iteration
            mis_index = (y * X_b.dot(self.theta) <= 0).flatten()
            X_b_mis = X_b[mis_index]
            if len(X_b_mis) == 0:
                # no misclassified samples any more
                break
            y_mis = y[mis_index]
            print('epoch: {}. loss (distance of misclassified samples): {}'.format(epoch,
                                                                                   self._loss(X_b_mis, y_mis)))

            # update theta
            self.theta += self.learning_rate * np.expand_dims(np.sum(y_mis * X_b_mis, axis=0), axis=1)
            del X_b_mis, y_mis, mis_index

    def predict(self, X):
        X_b = np.insert(X, 0, 1, axis=1)
        return np.sign(X_b.dot(self.theta))

    def _loss(self, X_mis, y_mis):
        return -np.sum(y_mis * (X_mis.dot(self.theta)))


if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    dataset = pd.read_csv('../dataset/mnist.csv', header=0).values

    X = dataset[:, 1:]
    y = dataset[:, 0]
    y[y == 0] = -1
    y = np.expand_dims(y, axis=1).astype(np.float64)

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=40)

    p = Perceptron(learning_rate=0.1e-4, n_epoch=20)
    p.fit(train_X, train_y)

    pred = p.predict(test_X)
    print("accuracy: ", np.sum(pred == test_y)/len(test_y))
