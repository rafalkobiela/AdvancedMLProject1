import numpy as np


def precision_at_10(y_true, y_pred):

    if len(y_true) != len(y_pred):
        raise Exception("Lengths of y_test and probabilities vector are different!")

    sorted_index = np.argsort(y_pred)[::-1]

    y_test_sorted = y_true[sorted_index][:round(len(y_true) * 0.1) - 1]

    return sum(y_test_sorted) / len(y_test_sorted)










