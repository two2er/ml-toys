import numpy as np


class SVM:
    """
    Tong Ji Xue Xi Fang Fa (Statistical Learning Methods) by Hang Li, Tsinghua University Press, 2012
    Chapter 7.
    http://www.hangli-hl.com/books.html
    """
    def __init__(self, kernel='linear', epsilon=1e-3, max_iter=1000, C=100):
        """ non-linear support vector machine
        :param kernel: kernel function
        :param epsilon: SMO KKT condition check precision
        :param max_iter: SMO max number of iterations
        :param C: penalty parameter (soft margin)
        """
        if kernel == 'linear':
            self.kernel = self._linear
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.C = C

    def _linear(self, prod):
        return prod

    def fit(self, X, y):
        self.X, self.y = X, y
        self.n_samples, self.n_features = X.shape

        # Lagrange multipliers
        self.alpha = np.zeros(self.n_samples, dtype=np.float64)
        # bias
        self.b = 0.
        # predictions of X
        self.g = np.zeros(self.n_samples, dtype=np.float64)
        # E: g - y
        self.E = self.g - self.y
        # K[i][j]: kernel(x_i.dot(x_j))
        self.K = self.kernel(X.dot(X.T))

        for k in range(self.max_iter):
            # select two multipliers optimally
            i1, i2 = self._select_alpha()
            if i1 < 0:     # all alphas satisfy KKT
                print('satisfy KKT')
                break
            # update alpha_1 and alpha_2
            alpha_1, alpha_2 = self.alpha[i1], self.alpha[i2]
            E_1, E_2 = self.E[i1], self.E[i2]
            y_1, y_2 = self.y[i1], self.y[i2]
            K_11, K_22, K_12 = self.K[i1][i1], self.K[i2][i2], self.K[i1][i2]
            eta = K_11 + K_22 - 2 * K_12
            alpha_2_new = alpha_2 + y_2 * (E_1 - E_2) / eta
            # quadratic programming cut
            L = max(0, alpha_2 - alpha_1) if y_1 != y_2 else max(0, alpha_2 + alpha_1 - self.C)
            H = min(self.C, self.C + alpha_2 - alpha_1) if y_1 != y_2 else min(self.C, alpha_2 + alpha_1)
            if alpha_2_new > H:
                alpha_2_new = H
            elif alpha_2_new < L:
                alpha_2_new = L
            alpha_1_new = alpha_1 + y_1 * y_2 * (alpha_2 - alpha_2_new)
            # update bias b
            b_1_new = -E_1 - y_1 * K_11 * (alpha_1_new - alpha_1) \
                      - y_2 * K_12 * (alpha_2_new - alpha_2) + self.b
            b_2_new = -E_2 - y_1 * K_12 * (alpha_1_new - alpha_1) \
                      - y_2 * K_22 * (alpha_2_new - alpha_2) + self.b
            b_new = (b_1_new + b_2_new) / 2

            # update g and E
            """
            TODO: don't know why, updating g/E of i1 and i2 but not all alphas, does decrease error
            better than updating all alphas
            """
            # for i in range(len(self.g)):
            for i in [i1, i2]:
                self.g[i] = self.g[i] + (alpha_1_new - alpha_1) * y_1 * self.K[i1][i] + (alpha_2_new - alpha_2) \
                            * y_2 * self.K[i2][i] + b_new - self.b
                self.E[i] = self.g[i] - self.y[i]

            # update parameters
            self.alpha[i1] = alpha_1_new
            self.alpha[i2] = alpha_2_new
            self.b = b_new

            # training log
            print('epoch: {}. loss (mse): {}'.format(k, self._loss(self.g, self.y)))

    @staticmethod
    def _loss(pred, y):
        # mse
        return np.sum((pred - y) ** 2) / len(y)

    def _select_alpha(self):
        # select two Lagrange multipliers in Heuristic way
        # alpha_1_list = [i for i in range(self.n_samples) if 0 < self.alpha[i] < self.C] \
        #     + [i for i in range(self.n_samples) if self.alpha[i] == 0 or self.alpha[i] == self.C]

        alpha_1_list = list(zip(range(self.n_samples), self.alpha))
        # check alpha between 0 and C first, then alpha = 0 or alpha = C
        alpha_1_list.sort(key=lambda x: np.random.randint(0, 5) if 0 < x[1] < self.C else np.random.randint(6, 10))
        for alpha_1_idx, alpha_1 in alpha_1_list:
            # whether the sample (X[alpha_1_idx], y[alpha_1_idx]) fulfills KKT condition
            y_g = self.y[alpha_1_idx] * self.g[alpha_1_idx]
            if not self._satisfy_KKT(y_g, alpha_1):
                # select the second alpha. maximize |E_1 - E_2|
                alpha_2_idx = np.argmax(np.abs(self.E - self.E[alpha_1_idx]))
                return alpha_1_idx, alpha_2_idx
        # every alpha satisfies KKT condition
        return -1, -1

    def _satisfy_KKT(self, y_g, alpha):
        if abs(alpha) < self.epsilon:  # alpha == 0
            return y_g >= 1
        elif abs(alpha - self.C) < self.epsilon:  # alpha == C
            return y_g <= 1
        else:  # 0 < alpha < C
            return abs(y_g - 1) < self.epsilon

    def predict(self, X):
        K = self.kernel(self.X.dot(X.T))
        return (self.alpha * self.y).dot(K) + self.b


def generate_dataset(n_samples=2000, n_features=10, scale=10, noise=False):
    # generate linear separable dataset
    X = np.random.random((n_samples, n_features)) * scale - scale / 2
    W = np.random.random((n_features, 1)) * scale - scale / 2
    b = np.random.rand() * scale - scale / 2
    y = X.dot(W) + b
    y = np.sign(y - np.mean(y)).flatten()

    if noise:   # add some noise
        rand_idx = np.random.choice(n_samples, int(n_samples*0.01))
        y[rand_idx] = -y[rand_idx]
    return X, y


if __name__ == '__main__':
    X, y = generate_dataset(n_samples=2000, n_features=10, noise=True)
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=40)
    model = SVM(max_iter=1000)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print('accuracy:', np.sum(np.sign(pred)==test_y) / len(test_y))
