import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import helper


def subtask1_predict(df, df_test, labels):
    # subtask 1
    ind_tsk1 = ['pid', 'BaseExcess', 'Fibrinogen', 'AST', 'Alkalinephos', 'Bilirubin_total', 'Lactate', 'TroponinI',
                'SaO2',
                'Bilirubin_direct', 'EtCO2']
    df_tsk1 = df.loc[:, ind_tsk1]

    labels_tsk1 = labels.loc[:, 'LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos',
                  'LABEL_Bilirubin_total',
                  'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
                  'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
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
    rand_indx = helper.random_gen(0, 9, down_size)
    X_train_tsk1, X_test_tsk1 = list(), list()
    X_train_tsk1 = np.asarray(helper.transform_rand_list(rand_indx, train_chunks, X_train_tsk1))
    X_test_tsk1 = np.asarray(helper.transform_rand_list(rand_indx, test_chunks, X_test_tsk1))

    # X_train_tsk1 = X_train_tsk1.reshape(-1, X_train_tsk1.shape[-1])

    # 18955 * 5 * 10
    med_test_svm = SVC(kernel='sigmoid', gamma='auto', probability=True, C=0.1)

    labels_one = np.ones((X_test_tsk1.shape[0], labels_sorted_tsk1.shape[1]))
    # labels_zero = np.ones((X_test_tsk1.shape[0], labels_sorted.shape[1]))
    # 18995, 5, 10

    for i in range(X_train_tsk1.shape[2]):
        med_test_svm.fit(X_train_tsk1[:, :, i], labels_sorted_tsk1[:, i])
        # labels_zero[:, i] = med_test_svm.predict_proba(X_test_tsk1[:, :, i])[:, 0]
        labels_one[:, i] = med_test_svm.predict_proba(X_test_tsk1[:, :, i])[:, 1]

    # print(labels_zero[1,:])
    # [0.73177289 0.92629629 0.76099499 0.76377979 0.75994957 0.79959082
    #  0.8998262  0.76630676 0.96553518 0.93355658]

    return None





