# -*- endoding: utf-8 -*-
import numpy as np

class LinearRegressor(object):

    def __init__(self, method='gradient', gradient='batch'):
        # 'gradient' method: gradient descent
        # 'normal' method: normal equation
        assert method == 'gradient' or method == 'normal', \
            'the method parameter should be "gradient" or "normal"'
        
        self.method = method
        # 'batch' gradient: batch gradient descent
        # 'stochastic' gradient: stochastic gradient descent
        # 'mini-batch' gradient: mini-batch gradient descent
        self.gradient = gradient
        self.theta = None

    def fit(self, X, y, eta=1e-2, epsilon=1e-3, batch_size=20, debug=False):
        # eta: learning rate
        # epsilon: tolerance
        # batch_size: batch size for mini-batch
        assert type(eta) == type(1e-1) or type(eta) == type(1), 'eta must be a number'
        assert len(X) == len(y), 'the amount of samples must be equal to the amount of labels'
        assert type(batch_size) == type(1), 'batch_size must be an int'

        X_b = np.c_[np.ones((len(X), 1)), np.array(X)]
        y = np.array(y)
        m, n = X_b.shape

        if self.method == 'normal':
            self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        if self.method == 'gradient':
            self.theta = np.random.rand(n, 1)
            
            if self.gradient == 'batch':
                self.batch_gradient(X_b, y, eta, epsilon, debug)
            elif self.gradient == 'stochastic':
                self.stochastic_gradient(X_b, y, eta, epsilon, debug)
            elif self.gradient == 'mini-batch':
                self.mini_batch(X_b, y, eta, epsilon, batch_size, debug)
            else:
                raise AssertionError('gradient parameter must be "batch", "stochastic" or "mini-batch"')
                        
    def batch_gradient(self, X, y, eta, epsilon, debug):
        m, n = X.shape
        epoch_count = 0
        while True:
            epoch_count += 1
            gradient_vector = 2 / m * X.T.dot(X.dot(self.theta) - y)
            if np.linalg.norm(gradient_vector) < epsilon:
                if debug:
                    print('end after', epoch_count,
                          'epochs. norm of gradient vector:', np.linalg.norm(gradient_vector))
                break
            self.theta = self.theta - eta * gradient_vector

    def stochastic_gradient(self, X, y, eta, epsilon, debug):
        m, n = X.shape
        epoch_count = 0
        while True:
            # I don't know how to early-stop, so I evaluate the norm of gradient vector
            # every 10 epochs. or terminate after 2000 epochs
            epoch_count += 1
            if epoch_count % 10 == 0:
                gradient_vector = 2 / m * X.T.dot(X.dot(self.theta) - y)
                if np.linalg.norm(gradient_vector) < epsilon:
                    if debug:
                        print('end after', epoch_count,
                            'epochs. norm of gradient vector:', np.linalg.norm(gradient_vector))
                    break
            if epoch_count == 2000:
                if debug:
                    print('end after', epoch_count,
                          'epochs. norm of gradient vector:', np.linalg.norm(gradient_vector))
                break
            # randomly shuffle X
            random_sequence = np.random.permutation([i for i in range(m)])
            eta = self.learning_schedule(eta)
            for i in range(m):
                xi = X[random_sequence[i]:random_sequence[i]+1]
                yi = y[random_sequence[i]:random_sequence[i]+1]
                gradient_vector = 2 * xi.T.dot(xi.dot(self.theta) - yi)
                self.theta = self.theta - eta * gradient_vector

    def learning_schedule(self, eta):
        # learning rate decay
        return eta * 0.99
    
    def mini_batch(self, X, y, eta, epsilon, batch_size, debug):
        m, n = X.shape
        epoch_count = 0
        while True:
            # I don't know how to early-stop, so I evaluate the norm of gradient vector
            # every 10 epochs. or terminate after 2000 epochs
            epoch_count += 1
            if epoch_count % 10 == 0:
                gradient_vector = 2 / m * X.T.dot(X.dot(self.theta) - y)
                #print(np.linalg.norm(gradient_vector))
                if np.linalg.norm(gradient_vector) < epsilon:
                    if debug:
                        print('end after', epoch_count,
                            'epochs. norm of gradient vector:', np.linalg.norm(gradient_vector))
                    break
            if epoch_count == 2000:
                if debug:
                    print('end after', epoch_count,
                          'epochs. norm of gradient vector:', np.linalg.norm(gradient_vector))
                break
            # randomly shuffle X
            random_sequence = np.random.permutation([i for i in range(m)])
            eta = self.learning_schedule(eta)
            for i in range((int)(m/batch_size)+1):
                begin = i * batch_size
                end = (i+1) * batch_size if (i+1) * batch_size < m else m
                xi = X[begin:end]
                yi = y[begin:end]
                gradient_vector = 2 * xi.T.dot(xi.dot(self.theta) - yi)
                self.theta = self.theta - eta * gradient_vector

    def predict(self, X):
        assert self.theta is not None, 'you must fit first then predict'
        assert np.array(X).shape[1] == len(self.theta) - 1, 'input size unmatched'

        X = np.c_[np.ones((len(X), 1)), X]
        
        return X.dot(self.theta)


    