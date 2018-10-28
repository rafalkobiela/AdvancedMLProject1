import numpy as np


def precision_at_10(y_test, pred):

    if len(y_test) != len(pred) or \
       len(y_test.shape) != 1 or \
       len(pred.shape) != 1:
        raise Exception("Lengths of y_test and probabilities vector are different!")

    sorted_index = np.argsort(pred)[::-1]

    y_test_sorted = y_test[sorted_index][:round(len(y_test)*0.1) -1]

    return sum(y_test_sorted) / len(y_test_sorted)










