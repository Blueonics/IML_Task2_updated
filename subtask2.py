import numpy as np
from sklearn.model_selection import RepeatedKFold
import helper
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import pandas as pd


def subtask2_predict(df, df_test, labels):
    # subtask 1
    labels_tsk2 = labels.loc[:, 'LABEL_Sepsis']
    labels_tsk2 = np.asarray(labels_tsk2)
    labels_sorted_tsk2 = labels_tsk2.reshape((labels_tsk2.shape[0], 1))

    train_chunks = helper.make_chunks(df.values)
    test_chunks = helper.make_chunks(df_test.values)

    #
    X_train_tsk2, X_test_tsk2 = list(), list()
    X_train_tsk2 = np.asarray(helper.transform_list(train_chunks, X_train_tsk2))
    X_test_tsk2 = np.asarray(helper.transform_list(test_chunks, X_test_tsk2))

    # print(X_train_tsk2.shape) #18995 * 12 * 2
    X_train_flatten = X_train_tsk2.reshape(X_train_tsk2.shape[0], X_train_tsk2.shape[1] * X_train_tsk2.shape[2]) # 18995 x (12 * number of features)
    X_test_flatten = X_test_tsk2.reshape(X_test_tsk2.shape[0], X_test_tsk2.shape[1] * X_test_tsk2.shape[2])

    # print(X_train_flatten.shape)
    # print(X_test_flatten.shape)

    nn_pca = PCA(n_components=6)
    X_train_pca = nn_pca.fit_transform(X_train_flatten)
    X_test_pca = nn_pca.transform(X_test_flatten)

    # print(X_train_pca.shape)
    # print(X_test_pca.shape)

    # DNN
    # cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
    # num_in, num_out = X_train_pca.shape[1], 1
    # accuracy = list()
    #
    # for train_idx, test_idx in cv.split(X_train_pca):
    #     X_train, X_test = X_train_pca[train_idx], X_train_pca[test_idx]
    #     labels_train, labels_test = labels[train_idx], labels[test_idx]
    #     model = helper.get_nn(num_in, num_out)
    #     model.fit(X_train, labels_train, epochs=100)
    #     labels_pred = model.predict(X_test)
    #     temp_check = accuracy_score(labels, labels_pred)
    #     accuracy.append(temp_check)

    # print(np.mean(accuracy))
    num_in, num_out = X_train_pca.shape[1], 1
    model = helper.get_nn(num_in, num_out)
    model.fit(X_train_pca, labels_sorted_tsk2)
    labels_two = model.predict(X_test_pca)

    return labels_two





