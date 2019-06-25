import numpy as np


class LinearRegression:

    def __init__(self, learning_rate=1e-3, n_epoch=2000):
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch

    def fit(self, X, y):
        """fit model and return None.

        begin to train a model based on trainset X and target value set y.
        if the gradient descent method is mini-batch, the parameter batch_size would be taken as
        the size of one batch

        X: n_sample*n_feature, y: n_sample*1
        """
        # add x_0 = 1 for samples
        X_b = np.insert(X, 0, 1, axis=1)
        # theta: randomly initialized: [-1/sqrt(n), 1/sqrt(n)]
        limit = np.sqrt(X_b.shape[1])
        self.theta = np.random.uniform(-1/limit, 1/limit, (X_b.shape[1], 1))

        self._training_method(X_b, y)

    def _training_method(self, X, y):
        raise NotImplementedError()

    @staticmethod
    def _loss(pred, y):
        # mse
        return np.sum((pred-y)**2) / len(y)

    def predict(self, X):
        """predict and return the prediction.

        predict the target values of X
        """
        assert self.theta is not None, 'you must train a model before predicting'
        X_b = np.insert(X, 0, 1, axis=1)
        return X_b.dot(self.theta)


class NormalRegression(LinearRegression):
    """
    training with the Normal Equation
    hat_theta = (X.T.dot(X)).inverse.dot(X.T).dot(y)
    in fact NormalRegression should not inherit from LinearRegression because it does not have
    learning_rate and epoch attributes.
    """
    def _training_method(self, X, y):
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


class BatchGradientRegression(LinearRegression):

    def _training_method(self, X, y):
        for epoch in range(self.n_epoch):
            gradient_vector = 2 / X.shape[0] * X.T.dot(X.dot(self.theta) - y)
            self.theta = self.theta - self.learning_rate * gradient_vector

            print('epoch: {}. loss (mse): {}'.format(epoch, self._loss(X.dot(self.theta), y)))


class StochasticGradientRegression(LinearRegression):

    def _training_method(self, X, y):
        m = X.shape[0]
        for epoch in range(self.n_epoch):
            # randomly shuffle X
            random_sequence = np.random.permutation([i for i in range(m)])
            self.learning_rate = self.learning_schedule(self.learning_rate)
            for i in range(m):
                xi = X[random_sequence[i]:random_sequence[i]+1]
                yi = y[random_sequence[i]:random_sequence[i]+1]
                gradient_vector = 2 * xi.T.dot(xi.dot(self.theta) - yi)
                self.theta = self.theta - self.learning_rate * gradient_vector

            print('epoch: {}. loss (mse): {}'.format(epoch, self._loss(X.dot(self.theta), y)))

    @staticmethod
    def learning_schedule(learning_rate):
        # learning rate decay
        return learning_rate * 0.99


class MiniBatchGradientRegression(LinearRegression):

    def __init__(self, learning_rate = 1e-3, n_epoch = 2000, batch_size=20):
        self.batch_size = batch_size
        super(MiniBatchGradientRegression, self).__init__(learning_rate, n_epoch)

    def _training_method(self, X, y):
        m = X.shape[0]
        for epoch in range(self.n_epoch):
            self.learning_rate = self.learning_schedule(self.learning_rate)
            for i in range(int(m/self.batch_size)+1):
                begin = i * self.batch_size
                end = (i+1) * self.batch_size if (i+1) * self.batch_size < m else m
                xi = X[begin:end]
                yi = y[begin:end]
                gradient_vector = 2 * xi.T.dot(xi.dot(self.theta) - yi)
                self.theta = self.theta - self.learning_rate * gradient_vector

            print('epoch: {}. loss (mse): {}'.format(epoch, self._loss(X.dot(self.theta), y)))

    @staticmethod
    def learning_schedule(learning_rate):
        # learning rate decay
        return learning_rate * 0.99


if __name__ == '__main__':
    import pandas as pd
    dataset = pd.read_csv('../dataset/abalone').values
    X, y = dataset[:, 1:].astype(np.float64), dataset[:, 0]
    y[y != 'M'] = -1.
    y[y == 'M'] = 1.
    y = np.expand_dims(y, axis=1).astype(np.float64)
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=40)
    model = BatchGradientRegression(n_epoch=20)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print('mse:', np.sum((pred-test_y)**2) / len(test_y))


"""this is the very basic linear regression model. some defects bothered me while testing.
1. if the feature values are big, the MSE would be very big and even disturb the 
   process of training. Sometimes it is useless to aply normalization to the trainset,
   because this model is very sensitive to the offset of data.
   At first, I compare the predicted weights to judge the performance of a model. However,
   in the case that feature values are much bigger than weights, the prediction of
   weights is strongly different from the true values, especially the offset, but
   MSE is not as obvious as the bias of predicted weights.
2. it is hard to find the appropriate learning rate and amount of epochs, therefore
   sometimes the outcome would be strange. If you print the MSE or l2-norm of gradient
   vector every gradient descent step, you would find that the number is 'exploding',
   which means it increases in a very high speed, and just after several iterations,
   it would become a nan.
"""
