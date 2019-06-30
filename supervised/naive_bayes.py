import numpy as np
from collections import Counter


class NaiveBayesClassifier:
    """Naive Bayes
        y = argmax_c P(Y=c)\prod_i P(X_i = x_i | Y=c)
        P(Y=c) is the priori probability that a label is of class c
        P(X_i = x_i | Y=c) is the posterior probability at Y=c that the ith feature
        is value x_i
        P(X_i = x_i) is assumed to obey Gaussian Distribution
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        # calculate P(Y=c)
        self.P_y_c = Counter(y)
        N = len(y)
        for key in self.P_y_c.keys():
            self.P_y_c[key] /= N
        # calculate P(X_i = x_i | Y=c), just get mean and var
        self.X_params = dict()
        for label_class in self.P_y_c.keys():
            self.X_params[label_class] = []
            X_posterior = X[y == label_class]
            for f in range(X_posterior.shape[1]):
                mean = np.mean(X_posterior[:, f])
                var = np.var(X_posterior[:, f])
                self.X_params[label_class].append((mean, var))

    def predict(self, X):
        return [self._predict_each(x) for x in X]

    def _predict_each(self, x):
        # calculate P(Y=c)\sum_i P(X_i = x_i | Y=c) for every c
        joints = []
        classes = list(self.P_y_c.keys())
        for label_class in classes:
            posterior = 1
            for i, value in enumerate(x):
                params = self.X_params[label_class][i]
                # log to avoid underflow
                posterior += np.log(self._Gaussian(value, *params))
            joint = self.P_y_c[label_class] * np.exp(posterior)
            joints.append(joint)
        return classes[np.argmax(joints)]

    @staticmethod
    def _Gaussian(value, mean, var):
        # add 1e-5 to avoid division by 0
        return 1/(np.sqrt(2*np.pi*var))*np.exp(-(value-mean)**2/(2*var+1e-5))


if __name__ == '__main__':
    import pandas as pd
    dataset = pd.read_csv('../dataset/abalone').values
    X, y = dataset[:, 1:].astype(np.float64), dataset[:, 0]
    y[y != 'M'] = -1.
    y[y == 'M'] = 1.
    y = y.astype(np.float64)
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=40)
    model = NaiveBayesClassifier()
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print('mse:', np.sum((pred-test_y)**2) / len(test_y))
