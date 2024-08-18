import numpy as np
from sklearn.tree import DecisionTreeClassifier
from utils import accuracy
import pandas as pd
from sklearn.utils import resample


def sampling(X, y):
    X_class1 = X[y == 1]
    X_class0 = X[y == 0]

    num_samples_to_adjust = (len(X_class0) - len(X_class1)) // 2

    # oversampling
    X_class1_oversampled = X_class1.sample(
        num_samples_to_adjust, replace=True)
    y_class1_oversampled = pd.Series(
        np.ones(num_samples_to_adjust), index=X_class1_oversampled.index)

    # undersampling
    X_class0_undersampled = X_class0.sample(
        len(X_class0) - num_samples_to_adjust)
    y_class0_undersampled = pd.Series(
        np.zeros(len(X_class0_undersampled)), index=X_class0_undersampled.index)

    X_combined = pd.concat(
        [X_class0_undersampled, X_class1, X_class1_oversampled])
    y_combined = pd.concat(
        [y_class0_undersampled, y[y == 1], y_class1_oversampled])

    return X_combined, y_combined


def bootstrap(X, Y, model, num_bootstrap_samples=10):
    n = len(X)
    accuracies = []

    for _ in range(num_bootstrap_samples):
        # Create a bootstrap sample
        indices = np.random.choice(range(n), size=n, replace=True)
        X_bootstrap = X[indices]
        Y_bootstrap = Y[indices]
        # Out-of-bag (OOB) samples
        oob_indices = list(set(range(n)) - set(indices))
        X_oob = X[oob_indices]
        Y_oob = Y[oob_indices]

        model.fit(X_bootstrap, Y_bootstrap)

        predictions = model.predict(X_oob)
        oob_accuracy = accuracy(Y_oob, predictions)
        accuracies.append(oob_accuracy)

    return np.array(accuracies)


def forward_feature_selection(X, y, model, max_features):
    selected_features = []
    remaining_features = list(X.columns)
    best_score_all = -float('inf')
    selected_score = -float('inf')
    best_features = None

    while len(selected_features) < max_features:
        best_score = -float('inf')
        best_pair = None
        for i in range(len(remaining_features)):
            candidate_features = selected_features + [remaining_features[i]]
            X_subset = X[candidate_features]
            model.fit(X_subset, y)
            predictions = model.predict(X_subset)
            score = accuracy(y, predictions)

            if score > best_score:
                best_score = score
                best_pair = remaining_features[i]
                selected_score = score

            if score > best_score_all:
                best_score_all = score
                best_features = candidate_features
        if best_pair is not None:
            selected_features.append(best_pair)
            remaining_features.remove(best_pair)
        else:
            break

    return selected_features, selected_score, best_features, best_score_all


def k_fold_cross_validation(model, X, y, k=5):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    fold_size = X.shape[0] // k
    folds_X = [X[i*fold_size:(i+1)*fold_size] for i in range(k)]
    folds_y = [y[i*fold_size:(i+1)*fold_size] for i in range(k)]

    scores = []

    for i in range(k):
        X_train = np.concatenate([folds_X[j] for j in range(k) if j != i])
        y_train = np.concatenate([folds_y[j] for j in range(k) if j != i])
        X_val = folds_X[i]
        y_val = folds_y[i]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        score = accuracy(y_val, y_pred)
        scores.append(score)

    return np.array(scores)


class Cascade:
    def __init__(self, estimators):
        self.estimators = estimators
        self.estimators_ = []
        self.scores = []

    def fit(self, X, y, X_t, y_t):
        X_train = X.copy()
        y_train = y.copy()
        X_test = X_t.copy()
        y_test = y_t.copy()
        for estimator in self.estimators:

            estimator.fit(X_train, y_train)
            self.estimators_.append(estimator)

            y_pred_train = estimator.predict_proba(X_train)
            y_pred_test = estimator.predict_proba(X_test)

            y_pred = estimator.predict(X_test)
            score = accuracy(y_test, y_pred)
            self.scores.append(score)

            X_train = np.column_stack((X_train, y_pred_train))
            X_test = np.column_stack((X_test, y_pred_test))

        return np.array(self.scores)

    def predict(self, X):
        X_test = X.copy()
        for i, estimator in enumerate(self.estimators_):
            if i < len(self.estimators)-1:
                y_pred_train = estimator.predict_proba(X_test)
                X_test = np.column_stack((X_test, y_pred_train))
            else:
                return estimator.predict(X_test)


class BaggingClassifier:
    def __init__(self, base_estimator, n_estimators=5):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators_ = []
        self.scores = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            X_resampled, y_resampled = resample(X, y, n_samples=n_samples)
            estimator = self.base_estimator()
            estimator.fit(X_resampled, y_resampled)
            self.estimators_.append(estimator)
            y_pred = estimator.predict(X_resampled)
            score = accuracy(y_resampled, y_pred)
            self.scores.append(score)
        return np.array(self.scores)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_estimators))
        for i, estimator in enumerate(self.estimators_):
            predictions[:, i] = estimator.predict(X)
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x.astype(int))), axis=1, arr=predictions)


class AdaBoost:
    def __init__(self, n_clf=50):
        self.n_clf = n_clf

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples
        self.clfs = []
        self.clf_weights = []

        for _ in range(self.n_clf):
            clf = DecisionTreeClassifier(max_depth=1)
            clf.fit(X, y, sample_weight=w)
            pred = clf.predict(X)
            err = np.sum(w * (pred != y)) / np.sum(w)
            clf_weight = 0.5 * np.log((1 - err) / err)
            w = w * np.exp(-clf_weight * y * pred)
            w = w / np.sum(w)
            self.clfs.append(clf)
            self.clf_weights.append(clf_weight)

    def predict_proba(self, X):
        clf_preds = np.array([clf.predict(X) for clf in self.clfs])
        return np.dot(self.clf_weights, clf_preds)

    def predict(self, X):
        clf_preds = np.array([clf.predict(X) for clf in self.clfs])
        return np.sign(np.dot(self.clf_weights, clf_preds))


class SVM():
    def __init__(self, learning_rate=0.001, no_of_iterations=1000, lambda_parameter=0.01):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        y_label = np.where(self.Y <= 0, -1, 1)

        for i in range(self.no_of_iterations):
            for index, x_i in enumerate(self.X):
                condition = y_label[index] * \
                    (np.dot(x_i, self.w) - self.b) >= 1
                if (condition == True):
                    dw = 2 * self.lambda_parameter * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_parameter * \
                        self.w - np.dot(x_i, y_label[index])
                    db = y_label[index]

                self.w = self.w - self.learning_rate * dw
                self.b = self.b - self.learning_rate * db

    def predict_proba(self, X):
        output = np.dot(X, self.w) - self.b
        return output

    def predict(self, X):
        output = np.dot(X, self.w) - self.b
        predicted_labels = np.sign(output)
        y_hat = np.where(predicted_labels <= -1, 0, 1)
        return y_hat


class NonlinearSVM():
    def __init__(self, learning_rate=0.001, no_of_iterations=1000, lambda_parameter=0.01, gamma=0.1):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter
        self.gamma = gamma

    def rbf_kernel(self, X1, X2):
        result = np.exp(-self.gamma *
                        np.linalg.norm(X1[:, np.newaxis] - X2, axis=2) ** 2)
        return result

    def fit(self, X, Y):
        self.X = X
        self.Y = np.where(Y <= 0, -1, 1)
        self.m, self.n = X.shape
        self.alpha = np.zeros(self.m)
        self.b = 0

        self.K = self.rbf_kernel(self.X, self.X)

        for _ in range(self.no_of_iterations):
            for i in range(self.m):
                condition = self.Y[i] * \
                    (np.dot(self.alpha * self.Y, self.K[i]) + self.b) >= 1
                if condition:
                    self.alpha[i] -= self.learning_rate * \
                        (2 * self.lambda_parameter * self.alpha[i])
                else:
                    self.alpha[i] += self.learning_rate * \
                        (1 - self.Y[i] * (np.dot(self.alpha *
                         self.Y, self.K[i]) + self.b))
                    self.b += self.learning_rate * self.Y[i]

    def predict_proba(self, X):
        K = self.rbf_kernel(X, self.X)
        prediction = np.dot(K, self.alpha * self.Y) + self.b
        return prediction

    def predict(self, X):
        K = self.rbf_kernel(X, self.X)
        prediction = np.sign(np.dot(K, self.alpha * self.Y) + self.b)
        return np.where(prediction <= -1, 0, 1)


class Logistic:
    def __init__(self, n_iter=1000, lr=0.01):
        self.n_iter = n_iter
        self.lr = lr

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def predict_proba(self, X):
        X = self._add_intercept(X)
        h = self.sigmoid(np.dot(X, self.W))
        return h

    def predict(self, X):
        X = self._add_intercept(X)
        h = self.sigmoid(np.dot(X, self.W))
        return h.round()

    def fit(self, X, y):

        X = self._add_intercept(X)
        self.W = np.zeros(X.shape[1])

        for i in range(self.n_iter):
            z = np.dot(X, self.W)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.W -= self.lr * gradient
        return
