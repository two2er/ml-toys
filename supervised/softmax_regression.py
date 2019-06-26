import numpy as np


class SoftmaxRegression:
    """ softmax regression: a generalization of logistic regression

    http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/

    J(theta) = -[sum_{i=1}^m sum_{k=1}^K 1{y^{(i)}=k}log P(y^{(i)}=k)]
    where P(y^{(i)}=k) = exp(theta^{(k)}.T.dot(x^{(i)})) / sum_{j=1}^k[exp(theta^{(j)}.T.dot(x^{(i)}))]

    J_delta(theta) = -1/m * X^T.dot(ONE-P)
    where ONE is a m*k matrix, ONE_{i, h} is 1{y^{(i)}=h} (i-th sample, h-th class)
          P   is a m*k matrix, P_{i, h} is P(y^{(i)}=h)
    if the shape of the input y is n_sample * n_class (one-hot), then ONE = y
    """
    def __init__(self, learning_rate=1e-3, n_epoch=2000):
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch

    def _P(self, X):
        P = np.exp(X.dot(self.theta))
        normal = np.sum(P, axis=1).reshape(-1, 1)
        return P/normal

    def _one_hot(self, n_sample, y):
        if y.shape == (n_sample, 1) or y.shape == (n_sample,):
            n_class = len(np.unique(y))
            return_y = np.zeros((n_sample, n_class))
            for sample, class_ in enumerate(y.reshape(-1)):
                return_y[sample][class_] = 1
            return return_y
        else:
            return y

    def fit(self, X, y):
        """X: n_sample*n_feature, y: n_sample*n_class"""
        # add x_0 = 1 for samples
        X_b = np.insert(X, 0, 1, axis=1)
        y = self._one_hot(X_b.shape[0], y)
        # theta: randomly initialized: [-1/sqrt(n), 1/sqrt(n)]
        limit = np.sqrt(X_b.shape[1])
        # theta: n_features * n_class
        self.theta = np.random.uniform(-1/limit, 1/limit, (X_b.shape[1], y.shape[1]))

        for epoch in range(self.n_epoch):
            self.theta -= self.learning_rate * 1/X_b.shape[0] * X_b.T.dot(self._P(X_b)-y)

            pred = self._P(X_b)
            print('epoch: {}. loss (mse): {}'.format(epoch, self._loss(pred, y)))

    def predict(self, X):
        X_b = np.insert(X, 0, 1, axis=1)
        P = self._P(X_b)
        return np.argmax(P, axis=1).reshape(-1, 1)

    def predict_proba(self, X):
        X_b = np.insert(X, 0, 1, axis=1)
        return self._P(X_b)

    def _loss(self, pred, y):
        # cross entropy: -[sum_{i=1}^m sum_{k=1}^K 1{y^{(i)}=k}log P(y^{(i)}=k)]
        return -np.sum(y * np.log(pred))


if __name__ == '__main__':
    import pandas as pd
    dataset = pd.read_csv('../dataset/abalone').values
    X, y = dataset[:, 1:].astype(np.float64), dataset[:, 0]
    y[y != 'M'] = 0
    y[y == 'M'] = 1
    y = np.expand_dims(y, axis=1).astype(np.int)
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=40)
    model = SoftmaxRegression(n_epoch=20)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print('mse: ', np.sum((pred-test_y)**2) / len(test_y))
