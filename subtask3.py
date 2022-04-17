import numpy as np
import pandas as pd
from sklearn.svm import SVR
import helper
from sklearn import linear_model


def subtask3_predict(X_train_procs, X_test_procs, labels):
    ind_tsk3 = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
    labels = np.asarray(labels.loc[:, ind_tsk3])

    regressor = SVR(kernel='rbf', gamma='auto')

    predictions = np.ones((12664, 10))
    for i in range(labels.shape[1]):
        print("index ", i)
        curr_label = labels[:, i]
        regressor.fit(X_train_procs, curr_label)
        predictions[:, i] = regressor.predict(X_test_procs)
        print(predictions[:, i])

    return predictions

