import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import helper
import csv


def subtask1_predict(df, df_test, labels):
    # subtask 1
    ind_tsk1 = ['pid', 'BaseExcess', 'Fibrinogen', 'AST', 'Alkalinephos', 'Bilirubin_total', 'Lactate', 'TroponinI', 'SaO2', 'Bilirubin_direct', 'EtCO2']
    df_tsk1 = df.loc[:, ind_tsk1]

    labels_tsk1 = labels.loc[:, 'LABEL_BaseExcess':'LABEL_EtCO2']

    labels_sorted_tsk1 = np.asarray(labels_tsk1)

    # Imputer
    imputer = IterativeImputer(missing_values=np.nan, initial_strategy='median')
    #
    imputer.fit(df_tsk1)

    df_tsk1 = pd.DataFrame(imputer.fit_transform(df_tsk1))
    test_tsk1 = df_test.loc[:, ind_tsk1]
    test_tsk1 = pd.DataFrame(imputer.fit_transform(test_tsk1))

    # len: 18995
    train_chunks = helper.make_chunks(df_tsk1.values)
    test_chunks = helper.make_chunks(test_tsk1.values)

    down_size = 5
    rand_indx = helper.random_gen(0, 5, down_size)
    X_train_tsk1, X_test_tsk1 = list(), list()
    X_train_tsk1 = np.asarray(helper.transform_rand_list(rand_indx, train_chunks, X_train_tsk1))
    X_test_tsk1 = np.asarray(helper.transform_rand_list(rand_indx, test_chunks, X_test_tsk1))

    # X_train_tsk1 = X_train_tsk1.reshape(-1, X_train_tsk1.shape[-1])

    # 18955 * 5 * 10
    med_test_svm = SVC(kernel='sigmoid', gamma='auto', probability=True, C=0.1)

    labels_one = np.ones((X_test_tsk1.shape[0], labels_sorted_tsk1.shape[1]))
    # labels_zero = np.ones((X_test_tsk1.shape[0], labels_sorted.shape[1]))
    # 18995, 5, 10
    # print(len(test_chunks))
    # f = open('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/submission.csv', 'w', newline='')
    # writer = csv.writer(f)
    # for j in range(labels_one.shape[0]):
    #     writer.writerow(labels_one[j, :])

    for i in range(X_train_tsk1.shape[2]):
        med_test_svm.fit(X_train_tsk1[:, :, i], labels_sorted_tsk1[:, i])
        # labels_zero[:, i] = med_test_svm.predict_proba(X_test_tsk1[:, :, i])[:, 0]
        labels_one[:, i] = med_test_svm.predict_proba(X_test_tsk1[:, :, i])[:, 1]



    # temp = np.unique(labels_chunks.keys())
    # print(temp)
    return labels_one





