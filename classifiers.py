import numpy as np
import random


class PerceptronClassifier:
    def __init__(self):
        pass

    def __perceptron_one_step(self, x, y, current_theta, current_theta_0) -> (np.ndarray, float):
        """
        Properly updates the classification parameter, theta and theta_0, on a
        single step of the perceptron algorithm.

        Args:
            feature_vector - A numpy array describing a single data point.
            label - The correct classification of the feature vector.
            current_theta - The current theta being used by the perceptron
                algorithm before this update.
            current_theta_0 - The current theta_0 being used by the perceptron
                algorithm before this update.

        Returns: A tuple where the first element is a numpy array with the value of
        theta after the current update has completed and the second element is a
        real valued number with the value of theta_0 after the current updated has
        completed.
        """
        if y * (np.dot(current_theta, x) + current_theta_0) <= 0:
            current_theta = current_theta + (y * x)
            current_theta_0 = current_theta_0 + y

        return current_theta, current_theta_0

    def perceptron(self, X, Y, T) -> (np.ndarray, float):
        """
        Runs the full perceptron algorithm on a given set of data. Runs T
        iterations through the data set, there is no need to worry about
        stopping early.

        NOTE: Please use the previously implemented functions when applicable.
        Do not copy paste code from previous parts.

        NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

        Args:
            feature_matrix -  A numpy matrix describing the given data. Each row
                represents a single data point.
            labels - A numpy array where the kth element of the array is the
                correct classification of the kth row of the feature matrix.
            T - An integer indicating how many times the perceptron algorithm
                should iterate through the feature matrix.

        Returns: A tuple where the first element is a numpy array with the value of
        theta, the linear classification parameter, after T iterations through the
        feature matrix and the second element is a real number with the value of
        theta_0, the offset classification parameter, after T iterations through
        the feature matrix.
        """
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
        """
        Runs the average perceptron algorithm on a given set of data. Runs T
        iterations through the data set, there is no need to worry about
        stopping early.

        NOTE: Please use the previously implemented functions when applicable.
        Do not copy paste code from previous parts.

        NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


        Args:
            feature_matrix -  A numpy matrix describing the given data. Each row
                represents a single data point.
            labels - A numpy array where the kth element of the array is the
                correct classification of the kth row of the feature matrix.
            T - An integer indicating how many times the perceptron algorithm
                should iterate through the feature matrix.

        Returns: A tuple where the first element is a numpy array with the value of
        the average theta, the linear classification parameter, found after T
        iterations through the feature matrix and the second element is a real
        number with the value of the average theta_0, the offset classification
        parameter, found after T iterations through the feature matrix.

        Hint: It is difficult to keep a running average; however, it is simple to
        find a sum and divide.
        """
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
        """
        Properly updates the classification parameter, theta and theta_0, on a
        single step of the Pegasos algorithm

        Args:
            feature_vector - A numpy array describing a single data point.
            label - The correct classification of the feature vector.
            L - The lambda value being used to update the parameters.
            eta - Learning rate to update parameters.
            current_theta - The current theta being used by the Pegasos
                algorithm before this update.
            current_theta_0 - The current theta_0 being used by the
                Pegasos algorithm before this update.

        Returns: A tuple where the first element is a numpy array with the value of
        theta after the current update has completed and the second element is a
        real valued number with the value of theta_0 after the current updated has
        completed.
        """
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
        """
        Runs the Pegasos algorithm on a given set of data. Runs T
        iterations through the data set, there is no need to worry about
        stopping early.

        For each update, set learning rate = 1/sqrt(t),
        where t is a counter for the number of updates performed so far (between 1
        and nT inclusive).

        NOTE: Please use the previously implemented functions when applicable.
        Do not copy paste code from previous parts.

        Args:
            feature_matrix - A numpy matrix describing the given data. Each row
                represents a single data point.
            labels - A numpy array where the kth element of the array is the
                correct classification of the kth row of the feature matrix.
            T - An integer indicating how many times the algorithm
                should iterate through the feature matrix.
            L - The lamba value being used to update the Pegasos
                algorithm parameters.

        Returns: A tuple where the first element is a numpy array with the value of
        the theta, the linear classification parameter, found after T
        iterations through the feature matrix and the second element is a real
        number with the value of the theta_0, the offset classification
        parameter, found after T iterations through the feature matrix.
        """
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
    def __init__(self):
        pass

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
        """
        A classification function that uses theta and theta_0 to classify a set of
        data points.

        Args:
            feature_matrix - A numpy matrix describing the given data. Each row
                represents a single data point.
                    theta - A numpy array describing the linear classifier.
            theta - A numpy array describing the linear classifier.
            theta_0 - A real valued number representing the offset parameter.

        Returns: A numpy array of 1s and -1s where the kth element of the array is
        the predicted classification of the kth row of the feature matrix using the
        given theta and theta_0. If a prediction is GREATER THAN zero, it should
        be considered a positive classification.
        """
        preds = (np.dot(X, theta.T)) + theta_0
        preds[preds > 0] = 1
        preds[preds <= 0] = -1
        return preds

    def classifier_accuracy(self, classifier, X_train, X_val, Y_train, Y_val, **kwargs) -> (float, float):
        """
        Trains a linear classifier and computes accuracy.
        The classifier is trained on the train data. The classifier's
        accuracy on the train and validation data is then returned.

        Args:
            classifier - A classifier function that takes arguments
                (feature matrix, labels, **kwargs) and returns (theta, theta_0)
            train_feature_matrix - A numpy matrix describing the training
                data. Each row represents a single data point.
            val_feature_matrix - A numpy matrix describing the validation
                data. Each row represents a single data point.
            train_labels - A numpy array where the kth element of the array
                is the correct classification of the kth row of the training
                feature matrix.
            val_labels - A numpy array where the kth element of the array
                is the correct classification of the kth row of the validation
                feature matrix.
            **kwargs - Additional named arguments to pass to the classifier
                (e.g. T or L)

        Returns: A tuple in which the first element is the (scalar) accuracy of the
        trained classifier on the training data and the second element is the
        accuracy of the trained classifier on the validation data.
        """
        T = kwargs.get('T')
        L = kwargs.get('L')


        if ('T' in kwargs) and ('L' in kwargs):
            theta, theta_0 = classifier(X_train, Y_train, T, L)
        elif 'T' in kwargs:
            theta, theta_0 = classifier(X_train, Y_train, T)

        Y_train_predicted = self.classify(X_train, theta, theta_0)
        X_val_predicted = self.classify(X_val, theta, theta_0)

        accuracy_train = self.accuracy(Y_train_predicted, Y_train)
        accuracy_val = self.accuracy(X_val_predicted, Y_val)

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
