import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
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


# Imputer
imputer = IterativeImputer(missing_values=np.nan, initial_strategy='median')
df_tsk1 = df.loc[:, ind]
imputer.fit(df_tsk1)
df_tsk1 = pd.DataFrame(imputer.fit_transform(df_tsk1))

test_tsk1 = df_test.loc[:, ind]
test_tsk1 = pd.DataFrame(imputer.fit_transform(test_tsk1))


# len: 18995
train_chunks = helper.make_chunks(df_tsk1.values)
test_chunks = helper.make_chunks(test_tsk1.values)
labels_arr = np.asarray(labels.loc[:, 'LABEL_BaseExcess':'LABEL_EtCO2'])

down_size = 5
# data pre-processing
rand_indx = helper.random_gen(0, 11, down_size)

# size: 12 x 37
# temp = train_chunks.get(1)

X_train_tsk1 = list()
X_test_tsk1 = list()

# for key in train_chunks.keys():
#     curr_rows = train_chunks.get(key)
#     # reduce_rows size: 5x37 and drop ID column
#     reduce_rows = curr_rows[rand_indx, 1:]
#     X_train_tsk1.append(reduce_rows)

X_train_tsk1 = helper.transform_list(rand_indx, train_chunks, X_train_tsk1)
X_test_tsk1 = helper.transform_list(rand_indx, test_chunks, X_test_tsk1)

X_train_tsk1 = np.asarray(X_train_tsk1)
X_test_tsk1 = np.asarray(X_test_tsk1)
### X_train_tsk1 = X_train_tsk1.reshape(-1, X_train_tsk1.shape[-1])


# 18955 * 5 * 10
med_test_svm = SVC(kernel='sigmoid')
for i in range(X_train_tsk1.shape[0]):
    med_test_svm.fit(np.transpose(X_train_tsk1[i, :, :]), labels_arr[i, :])
    labels_pred = med_test_svm.predict(X_test_tsk1)




# y_pred_tsk1 = np.ones((int(X_test_tsk1.shape[0]/split_size), total_features))
# med_test_svm = SVC(kernel='sigmoid')
# med_test_svm.fit(X_test_reshape, labels_reshape)
# y_pred_tsk1 = med_test_svm.predict(X_test_reshape)


# for i in range(total_features):
# med_test_svm.fit(new_X_train[0, :, :], labels[:, 0])
# y_pred_tsk1[:, i] = med_test_svm.predict(new_X_test[i, :, :])


# if __name__ == '__main__':
