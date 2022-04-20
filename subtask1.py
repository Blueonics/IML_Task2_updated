import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score



def subtask1_predict(X_train_procs, X_test_procs, labels):
    labels = labels.loc[:,
             ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
              'LABEL_Lactate',
              'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2']]

    labels_sorted_tsk1 = np.asarray(labels)

    X_train, X_val, y_train, y_val = train_test_split(X_train_procs, labels_sorted_tsk1)
    # validation set
    # for i in range(y_train.shape[1]):
    #     print("subtask 3 index ", i)
    #     curr_label = y_train[:, i]
    #     med_test_svm = SVC(kernel='rbf', probability=True, class_weight='balanced', gamma='auto')
    #     med_test_svm.fit(X_train, curr_label)
    #     y_hat = med_test_svm.predict(X_val)
    #     results = roc_auc_score(y_val[:, i], y_hat)
    #     print(results)

    labels_one = np.ones((12664, 10))
    for i in range(labels_sorted_tsk1.shape[1]):
        print("subtask 1 index ", i)
        curr_label = labels_sorted_tsk1[:, i]
        med_test_svm = SVC(kernel='rbf', probability=True, class_weight='balanced', gamma='scale')
        med_test_svm.fit(X_train_procs, curr_label)
        labels_one[:, i] = med_test_svm.predict_proba(X_test_procs)[:, -1]
        print(labels_one[:, i])

    return labels_one

