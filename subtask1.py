import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import helper
import csv


def subtask1_predict(df, df_test, labels):
    labels = labels.loc[:, ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
          'LABEL_Lactate',
          'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']]

    labels_sorted_tsk1 = np.asarray(labels)
    X_train_tsk1 = np.asarray(df)[:, 2:]
    X_test_tsk1 = np.asarray(df_test)[:, 2:]
    num_samp_train = int(X_train_tsk1.shape[0]/12)
    num_samp_test = int(X_test_tsk1.shape[0]/12)

    X_train_tsk1 = X_train_tsk1.reshape((num_samp_train, 12, -1))
    X_test_tsk1 = X_test_tsk1.reshape((num_samp_test, 12, -1))
    print(X_train_tsk1.shape) # 18995x12x35

    X_train_tsk1 = helper.imputer(X_train_tsk1, df)
    X_test_tsk1 = helper.imputer(X_test_tsk1, df_test)

    X_train_norm, mean_train, std_train = helper.batch_norm(X_train_tsk1)
    X_test_norm = helper.batch_norm(X_test_tsk1, mean_train, std_train)

    #  PCA
    X_train_flatten = X_train_tsk1.reshape(X_train_tsk1.shape[0], X_train_tsk1.shape[1] * X_train_tsk1.shape[2])
    X_test_flatten = X_test_tsk1.reshape(X_test_tsk1.shape[0], X_test_tsk1.shape[1] * X_test_tsk1.shape[2])

    nn_pca = PCA(n_components=140)
    nn_pca.fit(X_train_flatten)
    X_train_tsk1 = nn_pca.transform(X_train_flatten)
    X_test_tsk1 = nn_pca.transform(X_test_flatten)

    med_test_svm = SVC(kernel='rbf', probability=True)

    labels_one = np.ones((12664, 10))
    for i in range(labels_sorted_tsk1.shape[0]):
        print("index ", i)
        curr_label = labels_sorted_tsk1[:, i]
        med_test_svm.fit(X_train_tsk1, curr_label)
        labels_one[:, i] = med_test_svm.predict_proba(X_test_tsk1)[:, -1]
        print(labels_one[:,i])

    return labels_one
