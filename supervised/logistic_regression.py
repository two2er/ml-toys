import numpy as np


class LogisticRegression:
    """Logistic regression classifier

    hat_p = h_theta(x) = sigma(x.dot(theta)), where sigma is a sigmoid function:
        sigma(t) = 1 / (1 + exp(-t))
    and the binary prediction target label hat_y is:
        hat_y = 0 if hat_p < 0.5 else 1

    cost function J:
        J(theta) = -1/m * sum_{i=1}^m[y^i*log(hat_p^i) + (1-y^i)*log(1-hat_p^i)]
        (partial derivative of theta_j)
        J'(theta) = 1/m * sum_{i=1}^m(sigma(theta.T.dot(x^i))-y^i).dot(x_j^i)
    """
    def __init__(self, learning_rate=1e-3, n_epoch=2000):
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.theta = None

    @staticmethod
    def _sigma(t):
        return 1/(1+np.exp(-t))

    def _loss(self, pred_y, y):
        # cross entropy
        return -np.sum(y * np.log(pred_y) + (1 - y) * np.log(1 - pred_y)) / y.shape[0]

    def fit(self, X, y):
        """X: n_sample*n_feature, y: n_sample*n_class"""
        # add x_0 = 1 for samples
        X_b = np.insert(X, 0, 1, axis=1)
        # theta: randomly initialized: [-1/sqrt(n), 1/sqrt(n)]
        limit = np.sqrt(X_b.shape[1])
        self.theta = np.random.uniform(-1/limit, 1/limit, (X_b.shape[1], 1))

        for epoch in range(self.n_epoch):
            self.theta -= self.learning_rate \
                          * 1/X_b.shape[0] * X_b.T.dot(self._sigma(X_b.dot(self.theta))-y)

            pred = self._sigma(X_b.dot(self.theta))
            print('epoch: {}. loss (cross entropy): {}'.format(epoch, self._loss(pred, y)))

    def predict(self, X):
        X_b = np.insert(X, 0, 1, axis=1)
        return np.around(self._sigma(X_b.dot(self.theta))).astype(int)

    def predict_proba(self, X):
        X_b = np.insert(X, 0, 1, axis=1)
        return [[x[0], y[0]] for x, y in \
                zip(self._sigma(X_b.dot(self.theta)), 1-self._sigma(X_b.dot(self.theta)))]


if __name__ == '__main__':
    import pandas as pd
    dataset = pd.read_csv('../dataset/abalone').values
    X, y = dataset[:, 1:].astype(np.float64), dataset[:, 0]
    y[y != 'M'] = 0.
    y[y == 'M'] = 1.
    y = np.expand_dims(y, axis=1).astype(np.float64)
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=40)
    model = LogisticRegression(n_epoch=20)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print('mse:', np.sum((pred-test_y)**2) / len(test_y))
