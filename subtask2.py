import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import helper
import csv
from sklearn.decomposition import PCA
import pandas as pd

# multilabel classification


def subtask2_predict(df, df_test, labels):
    # subtask 1
    labels_tsk2 = labels.loc[:, 'LABEL_Sepsis']
    labels_tsk2 = np.asarray(labels_tsk2)
    labels_sorted_tsk2 = labels_tsk2.reshape((labels_tsk2.shape[0], 1))

    # test_ind = ['pid', 'BaseExcess', 'Fibrinogen']
    # df = df.loc[:, test_ind]
    # df_test = df_test.loc[:, test_ind]

    # Imputer
    imputer = IterativeImputer(missing_values=np.nan, initial_strategy='median')
    imputer.fit(df)

    df_tsk2 = pd.DataFrame(imputer.transform(df))
    test_tsk2 = pd.DataFrame(imputer.transform(df_test))

    train_chunks = helper.make_chunks(df_tsk2.values)
    test_chunks = helper.make_chunks(test_tsk2.values)

    #
    X_train_tsk2, X_test_tsk2 = list(), list()
    X_train_tsk2 = np.asarray(helper.transform_list(train_chunks, X_train_tsk2))
    X_test_tsk2 = np.asarray(helper.transform_list(test_chunks, X_test_tsk2))

    # print(X_train_tsk2.shape) #18995 * 12 * 2
    X_train_flatten = X_train_tsk2.reshape(X_train_tsk2.shape[0], X_train_tsk2.shape[1] * X_train_tsk2.shape[2]) # 18995 x (12 * number of features)
    X_test_flatten = X_test_tsk2.reshape(X_test_tsk2.shape[0], X_test_tsk2.shape[1] * X_test_tsk2.shape[2])

    # print(X_train_flatten.shape)
    # print(X_test_flatten.shape)

    nn_pca = PCA(n_components=5)
    X_train_pca = nn_pca.fit_transform(X_train_flatten)
    X_test_pca = nn_pca.transform(X_test_flatten)

    # print(X_train_pca.shape)
    # print(X_test_pca.shape)

    # DNN
    num_in, num_out = X_train_pca.shape[1], 1
    model = helper.get_nn(num_in, num_out)
    model.fit(X_train_pca, labels_sorted_tsk2)
    y_pred = model.predict(X_test_pca)
    y_pred = y_pred.round()

    # print(y_pred[100:200, 0])
    print(y_pred.shape)
    return None





