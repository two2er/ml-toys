from __future__ import print_function, division
import numpy as np
import math

def polynomial_features(X, degree):
    n_samples, n_features = np.shape(X)

    def index_combinations():
        # combinations_with_replacement('ABCD', 2)   ->   AA AB AC AD BB BC BD CC CD DD
        # n_features = 3, degree = 2 ->
        # return [(), (0,), (1,), (2,), (0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs
    
    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))
    
    # suppose feature: a, b, c. degree: 2 -> [1, a, b, c, aa, ab, ac, bb, bc, cc]: 2-nd degree combination
    # of all features with all samples
    # eg. X = np.array([[1,2,3],[4,5,6],[7,8,9]]), degree = 2
    # combinations: [(), (0,), (1,), (2,), (0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    # return: array([[ 1.,  1.,  2.,  3.,  1.,  2.,  3.,  4.,  6.,  9.],
    #                [ 1.,  4.,  5.,  6., 16., 20., 24., 25., 30., 36.],
    #                [ 1.,  7.,  8.,  9., 49., 56., 63., 64., 72., 81.]])
    # there are (n+d)!/(n!d!) kinds of combinations. where n = n_features, d = degree.
    for i, index_combs in enumerate(combinations):  
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new
    
def normalize(X, axis=-1, order=2):
    # i think axis should not be 1 but 0
    """ Normalize the dataset X """
    # atleast_1d: Convert inputs to arrays with at least one dimension.
    # https://stackoverflow.com/questions/50079622/numpy-np-sum-with-negative-axis
    # axis with negative values. -1 means the last axis. the last axis of a 2-d matrix is row.
    # l2 is an array of l2-norm of every sample. (if order=2)
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    # set l2-norm to 1 if it is 0 in case of being divided by a 0.
    l2[l2 == 0] = 1
    # every sample value is divided by the l2-norm of that sample.
    return X / np.expand_dims(l2, axis)

    # l2: [1,2,3] -> np.expand_dims(l2, -1) -> [[1], [2], [3]]
    # you can take a quick look at np.expand_dims by this:
    # >>> a.shape
    # (2, 3)
    # >>> np.expand_dims(a, axis=0).shape
    # (1, 2, 3) 
    # >>> np.expand_dims(a, axis=1).shape
    # (2, 1, 3)
    # >>> np.expand_dims(a, axis=2).shape
    # (2, 3, 1)


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
        # it should be (n_features, 1) but not (n_features, )
        self.w = np.random.uniform(-limit, limit, (n_features, 1))

    def fit(self, X, y):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self.initialize_weights(n_features=X.shape[1])

        # Do gradient descent for n_iterations
        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)
            # Calculate l2 loss
            mse = np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))
            self.training_errors.append(mse)
            # Gradient of l2 loss w.r.t w
            # X.T.dot(-(y - y_pred)), not -(y - y_pred).dot(X)
            grad_w = X.T.dot(-(y - y_pred)) + self.regularization.grad(self.w)
            # Update the weights
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred

class LinearRegression(Regression):
    """Linear model.
    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If 
        false then we use batch optimization by least squares.
    """
    def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        self.gradient_descent = gradient_descent
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(n_iterations=n_iterations,
                                            learning_rate=learning_rate)
    def fit(self, X, y):
        # If not gradient descent => Least squares approximation of w
        if not self.gradient_descent:
            # Insert constant ones for bias weights
            X = np.insert(X, 0, 1, axis=1)
            # Calculate weights by least squares (using Moore-Penrose pseudoinverse)
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = X_sq_reg_inv.dot(X.T).dot(y)
        else:
            super(LinearRegression, self).fit(X, y)

class LassoRegression(Regression):
    """Linear regression model with a regularization factor which does both variable selection 
    and regularization. Model that tries to balance the fit of the model with respect to the training 
    data and the complexity of the model. A large regularization factor with decreases the variance of 
    the model and do para.
    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    reg_factor: float
        The factor that will determine the amount of regularization and feature
        shrinkage. 
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    # why is polynomialization step added in this class?
    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01):
        self.degree = degree
        self.regularization = l1_regularization(alpha=reg_factor)
        super(LassoRegression, self).__init__(n_iterations, 
                                            learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super(LassoRegression, self).fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(LassoRegression, self).predict(X)

class PolynomialRegression(Regression):
    """Performs a non-linear transformation of the data before fitting the model
    and doing predictions which allows for doing non-linear regression.
    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, degree, n_iterations=3000, learning_rate=0.001):
        self.degree = degree
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(PolynomialRegression, self).__init__(n_iterations=n_iterations,
                                                learning_rate=learning_rate)

    def fit(self, X, y):
        X = polynomial_features(X, degree=self.degree)
        super(PolynomialRegression, self).fit(X, y)

    def predict(self, X):
        X = polynomial_features(X, degree=self.degree)
        return super(PolynomialRegression, self).predict(X)

class RidgeRegression(Regression):
    """Also referred to as Tikhonov regularization. Linear regression model with a regularization factor.
    Model that tries to balance the fit of the model with respect to the training data and the complexity
    of the model. A large regularization factor with decreases the variance of the model.
    Parameters:
    -----------
    reg_factor: float
        The factor that will determine the amount of regularization and feature
        shrinkage. 
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, reg_factor, n_iterations=1000, learning_rate=0.001):
        self.regularization = l2_regularization(alpha=reg_factor)
        super(RidgeRegression, self).__init__(n_iterations, 
                                            learning_rate)

class PolynomialRidgeRegression(Regression):
    """Similar to regular ridge regression except that the data is transformed to allow
    for polynomial regression.
    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    reg_factor: float
        The factor that will determine the amount of regularization and feature
        shrinkage. 
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01, gradient_descent=True):
        self.degree = degree
        self.regularization = l2_regularization(alpha=reg_factor)
        super(PolynomialRidgeRegression, self).__init__(n_iterations, 
                                                        learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super(PolynomialRidgeRegression, self).fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(PolynomialRidgeRegression, self).predict(X)

class ElasticNet(Regression):
    """ Regression where a combination of l1 and l2 regularization are used. The
    ratio of their contributions are set with the 'l1_ratio' parameter.
    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    reg_factor: float
        The factor that will determine the amount of regularization and feature
        shrinkage. 
    l1_ration: float
        Weighs the contribution of l1 and l2 regularization.
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, degree=1, reg_factor=0.05, l1_ratio=0.5, n_iterations=3000, 
                learning_rate=0.01):
        self.degree = degree
        self.regularization = l1_l2_regularization(alpha=reg_factor, l1_ratio=l1_ratio)
        super(ElasticNet, self).__init__(n_iterations, 
                                        learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super(ElasticNet, self).fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(ElasticNet, self).predict(X)


def mse(pred_y, y):
    return np.linalg.norm(pred_y-y)

def test_linear_regression():
    X = 2 * np.random.uniform(0, 1, (100, 1))
    #X = np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    X = np.c_[np.random.uniform(0, 1, (100, 1)), np.random.uniform(0, 5000, (100, 1)), np.random.uniform(0, 100, (100, 1))]
    y = 4 + X.dot(np.array([[300],[2000],[100]]))

    lr = LinearRegression(learning_rate=0.000000000001, n_iterations=5000)
    lr.fit(X, y)
    #test_X = np.c_[np.random.uniform(0, 1, (100, 1)), np.random.uniform(0, 50, (100, 1)), np.random.uniform(0, 100, (100, 1))]
    test_X = X
    # test_y = 4 + test_X.dot(np.array([[3],[2],[1]]))
    test_y = y
    pred_y = lr.predict(test_X)

    print(lr.w)
    print(mse(pred_y, test_y))

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