import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import helper

df = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/train_features.csv')
df_test = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/test_features.csv')
labels = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/train_labels.csv')
ind = ['pid', 'BaseExcess', 'Fibrinogen', 'AST', 'Alkalinephos', 'Bilirubin_total', 'Lactate', 'TroponinI', 'SaO2',
       'Bilirubin_direct', 'EtCO2']

# remove all zero rows from labels
labels = labels.loc[:, 'LABEL_BaseExcess':'LABEL_EtCO2']
# index_list = labels.index[(labels == 0).all(axis=1)].tolist()
index_list = labels.index[labels.values.sum(axis=1) == 0].tolist()

# labels = labels.loc[~(labels == 0).all(axis=1)]
# labels = labels.loc[labels.values.sum(axis=1) != 0]
labels_sorted = np.asarray(labels)
# index_list = np.array(np.where(np.all(labels_sorted == 0, axis=1)))

# print(len(index_list))

# print(labels_sorted.shape)


# Imputer
imputer = IterativeImputer(missing_values=np.nan, initial_strategy='mean')
df_tsk1 = df.loc[:, ind]
imputer.fit(df_tsk1)
df_tsk1 = pd.DataFrame(imputer.fit_transform(df_tsk1))
test_tsk1 = df_test.loc[:, ind]
test_tsk1 = pd.DataFrame(imputer.fit_transform(test_tsk1))

# len: 18995
train_chunks = helper.make_chunks(df_tsk1.values)
test_chunks = helper.make_chunks(test_tsk1.values)

down_size = 5
rand_indx = helper.random_gen(0, 11, down_size)
X_train_tsk1, X_test_tsk1 = list(), list()
X_train_tsk1 = np.asarray(helper.transform_list(rand_indx, train_chunks, X_train_tsk1))
X_test_tsk1 = np.asarray(helper.transform_list(rand_indx, test_chunks, X_test_tsk1))


### X_train_tsk1 = X_train_tsk1.reshape(-1, X_train_tsk1.shape[-1])

# 18955 * 10
med_test_svm = SVC(kernel='sigmoid', gamma='auto')

labels_pred = np.ones((X_test_tsk1.shape[0], labels_sorted.shape[1]))


X_train_tsk1 = np.swapaxes(X_train_tsk1, 1, 2) # 18995, 5, 10

# Method 1
for i in range(X_train_tsk1.shape[0]):
    if i in index_list:
        pass
    else:
        med_test_svm.fit(X_train_tsk1[i, :, :], labels_sorted[i, :])
        #labels_pred = med_test_svm.predict(np.transpose(X_test_tsk1[i, :, :]))
        # bug when we have all 1's labels or 0's labels