import numpy as np


class Network:
    """a naive neural network.
    my knowledge of Back Propagation mainly comes from this article:
    http://neuralnetworksanddeeplearning.com/chap2.html
    """
    def __init__(self, learning_rate=1e-2, epoch=10, batch_size=1, shape=None):
        """
        :param learning_rate:
        :param epoch:
        :param batch_size:
        :param shape: shape of the constructed network.
                      for example, shape = [4, 5, 3], which means except for the input
                      layer and the output layer, there are 3 intermediate layers, and
                      they 4, 5, 3 intermediate nodes respectively.
        """
        self.shape = shape
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size

        self.W = None
        self.b = None
        self.n_samples, self.n_features = -1, -1
        self.layer_num = -1

    def fit(self, X, Y):
        """
        :param X: X.shape is [n_samples, n_features]
        :param Y: y.shape is [1, n_samples]
        :return:
        """
        self.n_samples, self.n_features = X.shape

        self.shape.insert(0, self.n_features)  # input layer
        self.shape.append(1)  # output layer. currently only support 1 target
        self.layer_num = len(self.shape)
        # W for each layer. W for the first layer (input layer) is None
        self.W = [None for _ in range(self.layer_num)]
        # b for each layer. b for the first layer (input layer) is None
        self.b = [None for _ in range(self.layer_num)]
        # W[layer]: connecting (layer-1)-th and layer-th layer
        for layer in range(1, self.layer_num):
            self.W[layer] = np.zeros((self.shape[layer - 1], self.shape[layer]), dtype=np.float64)
            self.b[layer] = np.zeros((1, self.shape[layer]), dtype=np.float64)

        # begin learning
        for ep in range(self.epoch):
            # for x, y in zip(X, Y):
            #     nabla_W, nabla_b = self.backdrop(np.expand_dims(x, axis=0), np.expand_dims(y, axis=0))

            i = 0
            while self.batch_size * i <= self.n_samples:
                nabla_W, nabla_b = self.backdrop(X[i * self.batch_size:(i + 1) * self.batch_size],
                                                 Y[i * self.batch_size:(i + 1) * self.batch_size])

                # update
                for j in range(1, self.layer_num):
                    self.W[j] -= self.learning_rate * nabla_W[j]
                    self.b[j] -= self.learning_rate * nabla_b[j]

                i += 1

            # validate
            pred_Y = self.predict(X)
            print('round', ep, 'mse:', self._loss(pred_Y, Y))

    def predict(self, X):
        assert self.layer_num > 0, 'untrained'
        tmp_X = X
        for layer in range(1, self.layer_num):
            tmp_X = self.sigma(tmp_X.dot(self.W[layer]) + self.b[layer])
        return tmp_X

    @staticmethod
    def _loss(pred_Y, Y):
        return np.sum((pred_Y-Y)**2) / len(Y)

    def backdrop(self, x, y):
        # feed forward
        z_s, a_s = [None for _ in range(self.layer_num)], [None for _ in range(self.layer_num)]
        a_s[0] = x
        for layer in range(1, self.layer_num):
            z_s[layer] = a_s[layer-1].dot(self.W[layer]) + self.b[layer]
            a_s[layer] = self.sigma(z_s[layer])

        # backward
        delta_s = [None for _ in range(self.layer_num)]
        delta_s[-1] = self.nabla_C_a(a_s[-1], y) * self.sigma_prime(z_s[-1])
        for layer in range(self.layer_num - 2, 0, -1):
            delta_s[layer] = delta_s[layer+1].dot(self.W[layer+1].T) * self.sigma_prime(z_s[layer])

        # gradient
        nabla_W_s, nabla_b_s = [None for _ in range(self.layer_num)], [None for _ in range(self.layer_num)]
        for layer in range(1, self.layer_num):
            nabla_W_s[layer] = a_s[layer - 1].T.dot(delta_s[layer])
            nabla_b_s[layer] = np.sum(delta_s[layer], axis=0)

        return nabla_W_s, nabla_b_s

    def nabla_C_a(self, a, y):
        # partial derivative C/a
        # use MSE as cost function
        return 2 * (a - y)

    @staticmethod
    def sigma(z):
        # sigma: sigmoid function: 1 / (1 + e^{-x})
        return 1.0 / (1.0 + np.exp(-z))

    def sigma_prime(self, z):
        return self.sigma(z) * (1 - self.sigma(z))


if __name__ == '__main__':
    from sklearn.datasets import make_hastie_10_2

    from sklearn.model_selection import train_test_split
    X, y = make_hastie_10_2(random_state=42)
    y = np.expand_dims(y, axis=1)
    # Split into training and test set
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

    model = Network(epoch=200, shape=[2, 3, 4], batch_size=100, learning_rate=1e-3)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print('loss', model._loss(pred, test_y))
