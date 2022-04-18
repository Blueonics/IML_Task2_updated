import numpy as np
import helper
from sklearn.metrics import accuracy_score
import pandas as pd


def subtask2_predict(X_train_procs, X_test_procs, labels):
    labels_tsk2 = labels.loc[:, 'LABEL_Sepsis']
    labels_tsk2 = np.asarray(labels_tsk2)
    labels_sorted_tsk2 = labels_tsk2.reshape((labels_tsk2.shape[0], 1))

    num_in, num_out = X_train_procs.shape[1], 1
    model = helper.get_nn(num_in, num_out)
    model.fit(X_train_procs, labels_sorted_tsk2)
    labels_two = model.predict(X_test_procs)
    print(labels_two.shape)
    return labels_two





