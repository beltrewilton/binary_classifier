import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from classifiers import Classifier
import utils

df = pd.read_csv('./data/Big_AHR.csv', usecols=['rating', 'review_text'])
train_data, validate_data, test_data = np.split(df.sample(frac=1).values,  # random_state=42  # https://www.youtube.com/watch?v=5ZLtcTZP2js&t=83s
                                                [int(.6*len(df)), int(.8*len(df))])
dictionary = utils.unigram_words(train_data[:, 1])
break_point = 3

X_train = utils.extract_unigram_vectors(train_data[:, 1], dictionary)
X_validate = utils.extract_unigram_vectors(validate_data[:, 1], dictionary)
X_test = utils.extract_unigram_vectors(test_data[:, 1], dictionary)

Y_train = train_data[:, 0]
Y_validate = validate_data[:, 0]
Y_test = test_data[:, 0]

for Y in [Y_train, Y_validate, Y_test]:
    Y[Y <= break_point] = -1
    Y[Y > break_point] = 1

T = 10
L = 0.01

clf = Classifier(X_train, X_validate, Y_train, Y_validate, T=T)

pct_train_accuracy, pct_val_accuracy = \
   clf.classifier_accuracy(clf.perceptron)
print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))

avg_pct_train_accuracy, avg_pct_val_accuracy = \
   clf.classifier_accuracy(clf.average_perceptron)
print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))

avg_peg_train_accuracy, avg_peg_val_accuracy = \
   clf.classifier_accuracy(clf.pegasos, L=L)
print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))



data = (X_train, Y_train, X_validate, Y_validate)

# values of T and lambda to try
Ts = [1, 5, 10, 15, 25, 50]
Ls = [0.001, 0.01, 0.1, 1, 10]

pct_tune_results = clf.tune_perceptron(Ts, *data)
print('perceptron valid:', list(zip(Ts, pct_tune_results[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(pct_tune_results[1]), Ts[np.argmax(pct_tune_results[1])]))

avg_pct_tune_results = clf.tune_avg_perceptron(Ts, *data)
print('avg perceptron valid:', list(zip(Ts, avg_pct_tune_results[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(avg_pct_tune_results[1]), Ts[np.argmax(avg_pct_tune_results[1])]))

# fix values for L and T while tuning Pegasos T and L, respective
fix_L = 0.01
peg_tune_results_T = clf.tune_pegasos_T(fix_L, Ts, *data)
print('Pegasos valid: tune T', list(zip(Ts, peg_tune_results_T[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(peg_tune_results_T[1]), Ts[np.argmax(peg_tune_results_T[1])]))

fix_T = Ts[np.argmax(peg_tune_results_T[1])]
peg_tune_results_L = clf.tune_pegasos_L(fix_T, Ls, *data)
print('Pegasos valid: tune L', list(zip(Ls, peg_tune_results_L[1])))
print('best = {:.4f}, L={:.4f}'.format(np.max(peg_tune_results_L[1]), Ls[np.argmax(peg_tune_results_L[1])]))

utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
utils.plot_tune_results('Pegasos', 'T', Ts, *peg_tune_results_T)
utils.plot_tune_results('Pegasos', 'L', Ls, *peg_tune_results_L)

# Hyperparameters
# Tp = 10
# Lp = 0.01
# thetha_t, thetha_t_0 = clf.average_perceptron(X_train, Y_train, Tp)
# test_predicted_labels = clf.classify(X_test, thetha_t, thetha_t_0)
# test_accuracy = clf.accuracy(test_predicted_labels, Y_test)
# print("test bow accuracy: {}".format(test_accuracy))
#
# best_theta = thetha_t
# wordlist   = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
# sorted_word_features = clf.most_explanatory_word(best_theta, wordlist)
# print("Most Explanatory Word Features")
# print(sorted_word_features[:10])