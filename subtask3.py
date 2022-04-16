import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import helper
from sklearn import linear_model
from imblearn.over_sampling import SMOTE
from sklearn.multioutput import MultiOutputRegressor


def subtask3_predict(df, df_test, labels):
    # subtask 3
    ind_tsk3 = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

    labels = np.asarray(labels.loc[:, ind_tsk3])
    train_chunks = helper.make_chunks(df.values)
    test_chunks = helper.make_chunks(df_test.values)

    X_train_tsk3, X_test_tsk3 = list(), list()
    X_train_tsk3 = np.asarray(helper.transform_list(train_chunks, X_train_tsk3))
    X_test_tsk3 = np.asarray(helper.transform_list(test_chunks, X_test_tsk3))

    X_train_tsk3 = X_train_tsk3.mean(axis=1)
    X_test_tsk3 = X_test_tsk3.mean(axis=1)

    rig_model = linear_model.Ridge(alpha=0.1)
    multilabel_regressor = MultiOutputRegressor(rig_model, n_jobs=-1)

    # y_pred_tsk3 = np.ones((X_test_tsk3.shape[0], labels.shape[1]))
    multilabel_regressor.fit(X_train_tsk3, labels)

    y_pred_tsk3 = multilabel_regressor.predict(X_test_tsk3)
    y_pred_tsk3 = np.asarray(y_pred_tsk3)
    print(y_pred_tsk3.shape)

    return y_pred_tsk3

