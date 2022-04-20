import numpy as np
import helper
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split


def subtask2_predict(X_train_procs, X_test_procs, labels):
    labels_tsk2 = labels.loc[:, 'LABEL_Sepsis']
    labels_tsk2 = np.asarray(labels_tsk2)
    labels_sorted_tsk2 = labels_tsk2.reshape((labels_tsk2.shape[0], 1))

    num_in, num_out = X_train_procs.shape[1], 1
    model = helper.get_nn(num_in, num_out)
    weight = {0: 1.0, 1: 50.0}
    model.fit(X_train_procs, labels_sorted_tsk2, class_weight=weight, epochs=100, batch_size=10)
    labels_two = model.predict(X_test_procs)
    print(labels_two[0:100])


    # results = list()
    # # # define evaluation procedure
    # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # for train_ix, test_ix in cv.split(X_train_procs):
    #     X_train, X_test = X_train_procs[train_ix], X_train_procs[test_ix]
    #     y_train, y_test = labels_sorted_tsk2[train_ix], labels_sorted_tsk2[test_ix]
    #     model = helper.get_nn(num_in, num_out)
    #     model.fit(X_train, y_train, verbose=0, epochs=80, batch_size=10)
    #     yhat = model.predict(X_test)
    #     yhat = yhat.round()
    #     acc = accuracy_score(y_test, yhat)
    #     print('>%.3f' % acc)
    #     results.append(acc)
    # print(results)

    return labels_two