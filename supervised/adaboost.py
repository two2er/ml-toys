import numpy as np
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    def __init__(self, n_estimators=10, learning_rate=0.5, random_state=-1):
        """ Adaboost binary classifier. type of labels: 1 and -1
        ensemble n_estimators estimators: decision stump. or other classifiers that support
        sample weights.
        the weight of jth estimator is evaluated by:
        alpha_j = learning_rate * log((1-error_rate_j)/error_rate_j)
        
        W is updated on jth estimator by:
        w^{(i)} <- w^{(i)}*exp(alpha_j), if prediction_j^{(i)}!=y^{(i)}
        and then normalize: W_x = W_x / sum(W_x)
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        np.random.seed(random_state)
        self.estimators = []

    def fit(self, X, y):
        # initialize W
        W = np.full(X.shape[0], 1/X.shape[0])
        # estimator weight
        self.estimator_weight = np.empty(self.n_estimators)
        for i in range(self.n_estimators):
            estimator = DecisionTreeClassifier(max_depth=1, random_state=np.random.randint(10000))
            # train each estimator
            estimator.fit(X, y, sample_weight=W)
            pred = estimator.predict(X)
            error_rate = np.sum(W[pred!=y]) / np.sum(W)
            # estimator weight. adding an 1e-10 is to avoid being divided by 0
            self.estimator_weight[i] = self.learning_rate * \
                np.log((1-error_rate)/(1e-10+error_rate))
            # update W
            W *= np.exp(-self.estimator_weight[i] * y * pred)
            # normalize
            W /= np.sum(W)
            self.estimators.append(estimator)

            print('training: epoch {}. loss (error rate) = {}'.format(i, self._loss(self.predict(X), y)))

    def predict(self, X):
        """
        prediction(x) = argmax_k {sum_{j=1}^N alpha_j if prediction_j(x)=k}
        """
        pred = np.array([estimator.predict(X) for estimator in self.estimators])
        pred = pred * np.expand_dims(self.estimator_weight[:len(self.estimators)], axis=1)
        pred = np.sign(np.sum(pred, axis=0))
        return pred

    def _loss(self, pred, y):
        # error rate
        return np.sum(pred != y) / len(y)


if __name__ == '__main__':
    from sklearn.datasets import make_hastie_10_2

    from sklearn.model_selection import train_test_split
    X, y = make_hastie_10_2()
    # Split into training and test set
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

    model = AdaBoost(n_estimators=100, learning_rate=0.5, random_state=42)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print('accuracy', sum(pred==test_y)/len(test_y))
