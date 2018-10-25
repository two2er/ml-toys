from __future__ import print_function, division
import numpy as np
import math
"""
remember it is important to scale the data before performing most
regularized models
"""
class l1_regularization():
    """ 
    Regularization for Lasso Regression
    cost(theta) = MSE(theta) + alpha * sum(|theta_i|)
    subgradient_vector(theta, cost) = delta_theta MSE(theta) + alpha * sign(theta)
    note that the subscript of theta_i starts from 1, 
    which means that the weight of bias is not regularized.
    """
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        # is this a l2 regularization?
        return self.alpha * np.linalg.norm(w)

    def grad(self, w):
        # (sub)gradient_vector
        return self.alpha * np.sign(w)

class l2_regularization():
    """ 
    Regularization for Ridge Regression
    cost(theta) = MSE(theta) + alpha * 0.5 * sum(theta_i**2)
    gradient_vector(theta, cost) = delta_theta MSE(theta) + alpha * theta
    note that the subscript of theta_i starts from 1,
    which means that the weight of bias is not regularized.
    """
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * 0.5 *  w.T.dot(w)

    def grad(self, w):
        # (sub)gradient_vector
        return self.alpha * w

class l1_l2_regularization():
    """ 
    Regularization for Elastic Net Regression
    cost(theta) = MSE(theta) + r * alpha * sum(|theta_i|) + (1-r) * 0.5 * alpha * sum(theta_i**2)
    r = 1 -> l1 Lasso
    r = 0 -> l2 Ridge
    in the following, l1_ratio is equal to r.
    """
    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1_contr = self.l1_ratio * np.linalg.norm(w)
        l2_contr = (1 - self.l1_ratio) * 0.5 * w.T.dot(w) 
        return self.alpha * (l1_contr + l2_contr)

    def grad(self, w):
        l1_contr = self.l1_ratio * np.sign(w)
        l2_contr = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contr + l2_contr) 

class Regression(object):
    """ Base regression model. Models the relationship between a scalar dependent variable y and the independent 
    variables X. 
    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initialize_weights(self, n_features):
        """ Initialize weights randomly [-1/N, 1/N] """
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, 1))

    def fit(self, X, y):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        self.initialize_weights(n_features=X.shape[1])

        # Do gradient descent for n_iterations
        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)

            # Gradient of l2 loss w.r.t w
            grad_w = -X.T.dot(y - y_pred) + self.regularization.grad(self.w)
            # Update the weights
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred

class RidgeRegression(Regression):
   
    def __init__(self, reg_factor, n_iterations=1000, learning_rate=0.001):
        self.regularization = l2_regularization(alpha=reg_factor)
        super(RidgeRegression, self).__init__(n_iterations, 
                                            learning_rate)

def test_linear_regression():
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    lr = RidgeRegression(0.5)
    lr.fit(X, y)
    test_X = np.array([[0], [2]])
    pred_y = lr.predict(test_X)

    print(lr.w)
    print(pred_y)

    '''
    we can know that by setting parameter debug=True, stochastic and mini-batch gradient
    descent cost much less epochs to reach a relatively small norm of gradient vector, which
    indicates that stochastic and mini-batch train faster than batch gradient descent.
    '''
    
    # plt.plot(test_X, pred_y, 'r-')
    # plt.plot(X, y, 'b,')
    # plt.axis([0, 2, 0, 15])
    # plt.savefig('linear_regression.png')

if __name__ == '__main__':
    test_linear_regression()