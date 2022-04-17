import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from subtask1 import subtask1_predict
from subtask2 import subtask2_predict
from subtask3 import subtask3_predict
import helper
from zipfile import ZipFile
import csv

df = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/train_features.csv')
df_test = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/test_features.csv')
labels = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/train_labels.csv')

labels = labels.drop(['pid'], axis=1)

# preprocessing
X_train = np.asarray(df)[:, 2:]
X_test = np.asarray(df_test)[:, 2:]

hours = 12
num_samp_train = int(X_train.shape[0] / hours)
num_samp_test = int(X_test.shape[0] / hours)

X_train = X_train.reshape((num_samp_train, hours, -1))
X_test = X_test.reshape((num_samp_test, hours, -1))

X_train_imputed = helper.imputer(X_train, df)
X_test_imputed = helper.imputer(X_test, df_test)

X_train_norm, mean, std = helper.batch_norm(X_train_imputed)
X_test_norm = helper.batch_norm(X_test_imputed, mean, std)

# X_train_flatten = X_train_imputed.reshape((X_train_imputed.shape[0], X_train_imputed.shape[1] *
# X_train_imputed.shape[2])) X_test_flatten = X_test_imputed.reshape(X_test_imputed.shape[0], X_test_imputed.shape[1]
# * X_test_imputed.shape[2])

X_train_flatten = X_train_norm.reshape((X_train_norm.shape[0], X_train_norm.shape[1] * X_train_norm.shape[2]))
X_test_flatten = X_test_norm.reshape(X_test_norm.shape[0], X_test_norm.shape[1] * X_test_norm.shape[2])

nn_pca = PCA(n_components=120)
nn_pca.fit(X_train_flatten)
X_train_procs = nn_pca.transform(X_train_flatten)
X_test_procs = nn_pca.transform(X_test_flatten)

labels_one = subtask1_predict(X_train_procs, X_test_procs, labels)
labels_two = subtask2_predict(X_train_procs, X_test_procs, labels)
y_pred_three = subtask3_predict(X_train_procs, X_test_procs, labels)

extract_ID = np.array(df_test)
val, indices = np.unique(extract_ID[:, 0], return_index=True)
keys = np.asarray(extract_ID[:, 0][np.sort(indices)]).reshape((12664, 1))

arr1 = np.concatenate((labels_one, labels_two), axis=1)
values = np.concatenate((arr1, y_pred_three), axis=1)

lut = np.concatenate((keys, values), axis=1)

# write
f = open('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/submission/submission.csv', 'w', newline='')

header = ['pid', 'LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
          'LABEL_Lactate',
          'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2', 'LABEL_Sepsis', 'LABEL_RRate',
          'LABEL_ABPm', 'LABEL_SpO2',
          'LABEL_Heartrate']

writer = csv.DictWriter(f, delimiter=',', fieldnames=header)
writer.writeheader()

new_Writer = csv.writer(f, delimiter=',')
for i in lut:
    new_Writer.writerow(['{:.3f}'.format(x) for x in i])

with ZipFile('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/submission.zip', 'w') as zip:
    zip.write('submission/submission.csv')
