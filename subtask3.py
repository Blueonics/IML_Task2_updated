import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import helper
from sklearn import linear_model
import csv


def subtask3_predict(df, df_test, labels):
    # subtask 3
    ind_tsk3 = ['pid', 'RRate', 'ABPm', 'SpO2', 'Heartrate']
    df_tsk3 = df.loc[:, ind_tsk3]

    y_tsk3 = labels.loc[:, ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']]
    y_tsk3 = np.asarray(y_tsk3)

    imputer = IterativeImputer(missing_values=np.nan, initial_strategy='median')
    imputer.fit(df_tsk3)

    df_tsk3 = pd.DataFrame(imputer.fit_transform(df_tsk3))
    test_tsk3 = df_test.loc[:, ind_tsk3]
    test_tsk3 = pd.DataFrame(imputer.fit_transform(test_tsk3))
    test_ID = df_test.loc[:, 0]

    train_chunks = helper.make_chunks(df_tsk3.values)
    test_chunks = helper.make_chunks(test_tsk3.values)

    X_train_tsk3, X_test_tsk3 = list(), list()
    X_train_tsk3 = np.asarray(helper.transform_list(train_chunks, X_train_tsk3))
    X_test_tsk3 = np.asarray(helper.transform_list(test_chunks, X_test_tsk3))

    reg_model = linear_model.Ridge(alpha=0.1)

    y_pred_tsk3 = np.ones((X_test_tsk3.shape[0], y_tsk3.shape[1]))

    print(y_pred_tsk3.shape)
    for i in range(X_train_tsk3.shape[2]):
        reg_model.fit(X_train_tsk3[:, :, i], y_tsk3[:, i])
        y_pred_tsk3[:, i] = reg_model.predict(X_test_tsk3[:, :, i])

    f = open('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/submission.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerows()
    writer.writerows(y_pred_tsk3)

    return None
