""" adaboost. the base estimator is decision stump (binary decision tree with max depth = 1, which
    means it only has one internal node and two leaf nodes). 

we only consider binary classification problems, and the types of labels are +1 and -1.
"""

import numpy as np

class DecisionStump:

    def fit(self, X, y, Wx):
        """ train the model.

        Wx is the weights of samples in X. The classification error is calculated by:
        error = sum_{i that pred_i != y_i} Wx_i
        error_rate = error / sum_i Wx_i
        (however, Wx is always normalized, which means sum_i Wx_i is always equal to 1)
        """
        # find the best feature and feature value to split the root node
        best_feature, best_value, best_polarity, min_error_rate = -1, -1, 1, float('inf')

        # consider all features
        for feature in range(X.shape[1]):
            # consider all feature values of this feature
            for feature_value in np.unique(X[:, feature]):
                # By default, the label of left leaf node is +1 and the label of right leaf node is -1.
                # To reverse it, set polarity = -1.
                polarity = 1
                prediction = np.ones(len(y), dtype=np.int)
                prediction[X[:, feature] > feature_value] = -1
                # calculate error rate
                error_rate = np.sum(Wx[prediction!=y])
                # if error_rate > 0.5, reverse the polarity, which would makes the error_rate < 0.5
                if error_rate > 0.5:
                    polarity = -1
                    error_rate = 1 - error_rate

                if error_rate < min_error_rate:
                    best_feature, best_value, best_polarity = feature, feature_value, polarity
                    min_error_rate = error_rate

        # note down split feature, feature value and weighted error rate
        self.error_rate = min_error_rate
        self.split_feature = best_feature
        self.split_value = best_value
        self.polarity = best_polarity

    def predict(self, X):
        prediction = np.ones(X.shape[0])
        prediction[X[:, self.split_feature] > self.split_value] = -1
        # reverse the prediction if self.polarity == -1
        if self.polarity == -1:
            prediction = -prediction
        return prediction

class AdaBoost:
    def __init__(self, n_estimators=10, learning_rate=1e-3):
        """
        ensemble n_estimators estimators (decision stump) totally
        the weight of jth estimator is evaluated by:
        alpha_j = learning_rate * log((1-error_rate_j)/error_rate_j)
        
        Wx is updated on jth estimator by:
        w^{(i)} <- w^{(i)}*exp(alpha_j), if prediction_j^{(i)}!=y^{(i)}
        and then normalize: W_x = W_x / sum(W_x)
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = [DecisionStump() for _ in range(n_estimators)]

    def fit(self, X, y):
        # initialize Wx
        Wx = np.full(X.shape[0], 1/X.shape[0])
        # estimator weight
        self.estimator_weight = np.empty(self.n_estimators)
        for i, estimator in enumerate(self.estimators):
            # train each estimator
            estimator.fit(X, y, Wx)
            # estimator weight. 1e-10 is to avoid dividing 0
            self.estimator_weight[i] = self.learning_rate \
                               * np.log((1-estimator.error_rate)/(1e-10+estimator.error_rate))
            # update Wx, first predict with current estimator
            prediction = estimator.predict(X)
            Wx[prediction!=y] = Wx[prediction!=y] * np.exp(self.estimator_weight[i])
            # normalize
            Wx /= np.sum(Wx)

    def predict(self, X):
        """
        prediction(x) = argmax_k {sum_{j=1}^N alpha_j if prediction_j(x)=k}
        """
        preds = [estimator.predict(X) for estimator in self.estimators]
        prediction = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            pred_for_xi = np.array([preds[j][i] for j in range(self.n_estimators)])
            prediction[i] = np.sign(np.sum(pred_for_xi*self.estimator_weight))

        return prediction