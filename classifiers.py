import numpy as np
import random


class PerceptronClassifier:
    def __init__(self):
        pass

    def __perceptron_one_step(self, x, y, current_theta, current_theta_0) -> (np.ndarray, float):
        if y * (np.dot(current_theta, x) + current_theta_0) <= 0:
            current_theta = current_theta + (y * x)
            current_theta_0 = current_theta_0 + y

        return current_theta, current_theta_0

    def perceptron(self, X, Y, T) -> (np.ndarray, float):
        n, f = X.shape
        theta = np.zeros(f)
        theta_0 = 0
        for t in range(T):
            order = np.arange(n)
            np.random.shuffle(order)
            for i in order:
            # for i in self.get_order(n):
                theta, theta_0 = self.__perceptron_one_step(X[i], Y[i], theta, theta_0)

        return theta, theta_0

    def average_perceptron(self, X, Y, T) -> (np.ndarray, float):
        n, f = X.shape
        theta = np.zeros(f)
        theta_s = np.zeros(f)
        theta_0 = 0
        theta_0_s = 0
        for t in range(T):
            order = np.arange(n)
            np.random.shuffle(order)
            for i in order:
            # for i in self.get_order(n):
                theta, theta_0 = self.__perceptron_one_step(X[i], Y[i], theta, theta_0)
                theta_s += theta
                theta_0_s += theta_0

        return theta_s / (n * T), theta_0_s / (n * T)


class PegasosClassifier:
    def __init__(self):
        pass

    def __pegasos_one_step(self, x, y, L, eta, current_theta, current_theta_0) -> (np.ndarray, float):
        # current_theta = current_theta + (label * feature_vector)
        # current_theta_0 = current_theta_0 + label
        if y * (np.dot(current_theta, x) + current_theta_0) <= 1:
            current_theta = ((1 - (eta * L)) * current_theta) + (eta * y * x)
            # [optional]
            # current_theta = min(1, (1/np.sqrt(L)) / np.linalg.norm(current_theta)) * current_theta
            current_theta_0 = current_theta_0 + (eta * y)
        else:
            current_theta = (1 - (eta * L)) * current_theta

        return current_theta, current_theta_0

    def pegasos(self, X, Y, T, L) -> (np.ndarray, float):
        n, f = X.shape
        theta = np.zeros(f)
        theta_0 = 0
        counter = 1
        eta = 1 / np.sqrt(counter)
        for t in range(T):
            order = np.arange(n)
            np.random.shuffle(order)
            for i in order:
            # for i in self.get_order(n):
                theta, theta_0 = self.__pegasos_one_step(X[i], Y[i], L, eta, theta, theta_0)
                counter += 1
                eta = 1 / np.sqrt(counter)

        return theta, theta_0


class Classifier(PerceptronClassifier, PegasosClassifier):
    def __init__(self, X_train, X_val, Y_train, Y_val, T):
        self.X_train = X_train
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_val = Y_val
        self.T = T

    def get_order(self, n_samples):
        try:
            with open(str(n_samples) + '.txt') as fp:
                line = fp.readline()
                return list(map(int, line.split(',')))
        except FileNotFoundError:
            random.seed(1)
            indices = list(range(n_samples))
            random.shuffle(indices)
            return indices

    def accuracy(self, preds, targets) -> float:
        """
        Given length-N vectors containing predicted and target labels,
        returns the percentage and number of correct predictions.
        """
        return (preds == targets).mean()

    def classify(self, X, theta, theta_0) -> np.ndarray:
        preds = (np.dot(X, theta.T)) + theta_0
        preds[preds > 0] = 1
        preds[preds <= 0] = -1
        return preds

    def classifier_accuracy(self, classifier, L=None) -> (float, float):
        print("{} -> T: {}, L: {}".format(classifier, self.T, L))
        if self.T and L:
            theta, theta_0 = classifier(self.X_train, self.Y_train, self.T, L)
        elif self.T:
            theta, theta_0 = classifier(self.X_train, self.Y_train, self.T)

        Y_train_predicted = self.classify(self.X_train, theta, theta_0)
        X_val_predicted = self.classify(self.X_val, theta, theta_0)

        accuracy_train = self.accuracy(Y_train_predicted, self.Y_train)
        accuracy_val = self.accuracy(X_val_predicted, self.Y_val)

        return accuracy_train, accuracy_val

    def tune(self, train_fn, param_vals, train_feats, train_labels, val_feats, val_labels):
        train_accs = np.ndarray(len(param_vals))
        val_accs = np.ndarray(len(param_vals))

        for i, val in enumerate(param_vals):
            theta, theta_0 = train_fn(train_feats, train_labels, val)

            train_preds = self.classify(train_feats, theta, theta_0)
            train_accs[i] = self.accuracy(train_preds, train_labels)

            val_preds = self.classify(val_feats, theta, theta_0)
            val_accs[i] = self.accuracy(val_preds, val_labels)

        return train_accs, val_accs

    def tune_perceptron(self, *args):
        return self.tune(self.perceptron, *args)

    def tune_avg_perceptron(self, *args):
        return self.tune(self.average_perceptron, *args)

    def tune_pegasos_T(self, best_L, *args):
        def train_fn(features, labels, T):
            return self.pegasos(features, labels, T, best_L)

        return self.tune(train_fn, *args)

    def tune_pegasos_L(self, best_T, *args):
        def train_fn(features, labels, L):
            return self.pegasos(features, labels, best_T, L)

        return self.tune(train_fn, *args)

    def most_explanatory_word(self, theta, wordlist):
        """Returns the word associated with the bag-of-words feature having largest weight."""
        # theta(i) mientras mas grande es es mas explanatory_word / positive,
        # el indice asociado theta[i], wordlist[i] se mueve en par cuando se hace sort.
        return [word for (theta_i, word) in sorted(zip(theta, wordlist))[::-1]]
