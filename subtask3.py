import numpy as np
import pandas as pd
from sklearn.svm import SVR
import helper
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def subtask3_predict(X_train_procs, X_test_procs, y_train_procs):
    ind_tsk3 = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
    y_train_procs = np.asarray(y_train_procs.loc[:, ind_tsk3])
    print(y_train_procs.shape)
    regressor = SVR(kernel='rbf', gamma='scale')

    # X_train, X_val, y_train, y_val = train_test_split(X_train_procs, y_train_procs)
    # # validation set

    # for i in range(y_train.shape[1]):
    #     print("subtask 3 index ", i)
    #     curr_label = y_train[:, i]
    #     regressor.fit(X_train, curr_label)
    #     y_hat = regressor.predict(X_val)
    #     results = mean_squared_error(y_val[:, i], y_hat)
    #     print(results)

    predictions = np.ones((12664, y_train_procs.shape[1]))
    for i in range(y_train_procs.shape[1]):
        print("subtask 3 index ", i)
        curr_label = y_train_procs[:, i]
        regressor.fit(X_train_procs, curr_label)
        predictions[:, i] = regressor.predict(X_test_procs)
        # print(predictions[:, i])
    print(predictions.shape)

    return predictions

