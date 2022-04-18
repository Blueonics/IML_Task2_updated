import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import helper
import csv


def subtask1_predict(X_train_procs, X_test_procs, labels):
    labels = labels.loc[:, ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
          'LABEL_Lactate',
          'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']]

    labels_sorted_tsk1 = np.asarray(labels)

    med_test_svm = SVC(kernel='rbf', probability=True, gamma='auto')

    labels_one = np.ones((12664, 10))

    for i in range(labels_sorted_tsk1.shape[1]):
        print("subtask 1 index ", i)
        curr_label = labels_sorted_tsk1[:, i]
        med_test_svm.fit(X_train_procs, curr_label)
        labels_one[:, i] = med_test_svm.predict_proba(X_test_procs)[:, -1]
        print(labels_one[:,i])
    print(labels_one.shape)
    return labels_one
