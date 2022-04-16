import numpy as np
import pandas as pd
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputClassifier
import helper
import csv


def subtask1_predict(df, df_test, labels):

    labels = labels.loc[:, 'LABEL_BaseExcess':'LABEL_EtCO2']
    labels_sorted_tsk1 = np.asarray(labels)


    # # len: 18995
    train_chunks = helper.make_chunks(df.values)
    test_chunks = helper.make_chunks(df_test.values)

    # down_size = 5
    # rand_indx = helper.random_gen(0, 5, down_size)
    X_train_tsk1, X_test_tsk1 = list(), list()
    X_train_tsk1 = np.asarray(helper.transform_list(train_chunks, X_train_tsk1))
    X_test_tsk1 = np.asarray(helper.transform_list(test_chunks, X_test_tsk1))

    X_train_tsk1 = X_train_tsk1.mean(axis=1)
    X_test_tsk1 = X_test_tsk1.mean(axis=1)

    # upSampler = RandomOverSampler(sampling_strategy='minority')
    # downSampler = RandomUnderSampler(sampling_strategy=0.4)  # 40 percent majority more than minority
    # steps = [('o', upSampler), ('u', downSampler)]
    # pipe = Pipeline(steps=steps)
    #
    # X_train_tsk1, labels_sorted_tsk1 = pipe.fit_resample(X_train_tsk1, labels_sorted_tsk1)
    weights = {0:1.0, 1:100.0}
    med_test_svm = SVC(kernel='sigmoid', class_weight=weights, probability=True, gamma='scale', C=0.1)
    multilabel_classifier = MultiOutputClassifier(med_test_svm, n_jobs=-1)
    multilabel_classifier.fit(X_train_tsk1, labels_sorted_tsk1)

    labels_one = multilabel_classifier.predict_proba(X_test_tsk1)
    labels_one = np.transpose(np.asarray(labels_one)[:, :, 1]) # shape 10, 12664, 2
    print(labels_one.shape)

    return labels_one





