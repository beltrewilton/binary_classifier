import numpy as np
import pandas as pd
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
