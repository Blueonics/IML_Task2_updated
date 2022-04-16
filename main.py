import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.utils import resample
from subtask1 import subtask1_predict
from subtask2 import subtask2_predict
from subtask3 import subtask3_predict
from zipfile import ZipFile
import csv

df = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/train_features.csv')
df_test = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/test_features.csv')
labels = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/train_labels.csv')


header = ['pid', 'LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
          'LABEL_Lactate',
          'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct', 'LABEL_EtCO2', 'LABEL_Sepsis', 'LABEL_RRate',
          'LABEL_ABPm', 'LABEL_SpO2',
          'LABEL_Heartrate']

PID = np.asarray(df_test.loc[:, 'pid'].unique())
keys = PID.reshape((PID.shape[0], 1))

# temporary
df = df.loc[:, ['pid','AST','Alkalinephos']]
df_test = df_test.loc[:, ['pid', 'AST','Alkalinephos']]
labels = labels.loc[:, ['pid','LABEL_AST', 'LABEL_Alkalinephos']]

#
# labels = labels.drop(['pid',], axis=1)
# df = df.drop(['pid'], axis=1)
# df_test = df_test.drop(['pid'], axis=1)
#
# print(len(labels))
#
# num_O = len(labels[labels.values == 0])
# num_1 = len(labels[labels.values == 1])
# print("nm zero", num_O) #4361
# print("nm one", num_1) #1279
#
# zeros = labels[labels.values == 0]
# ones = labels[labels.values == 1]
#
# zero_downsample = resample(zeros, replace=True, n_samples=num_1, random_state=42)
#
# print("zero downsample shape", zero_downsample.shape) # 6496
#
# labels = pd.concat([zero_downsample, ones])
# # print("one shape", ones)
# # print("zero shape", zeros)
# print(len(labels))

# Impute
imputer = IterativeImputer(missing_values=np.nan, initial_strategy='mean')
imputer.fit(df)
df = pd.DataFrame(imputer.fit_transform(df))
df_test = pd.DataFrame(imputer.fit_transform(df_test))

labels = labels.drop(['pid'], axis=1)
# print(labels)

labels_one = subtask1_predict(df, df_test, labels)
# labels_two = subtask2_predict(df, df_test, labels)
# y_pred_three = subtask3_predict(df, df_test, labels)

#
# labels_one = np.ones((12664, 10))
labels_two = np.ones((12664, 1))
y_pred_three = np.ones((12664, 4))


arr1 = np.concatenate((labels_one, labels_two), axis=1)
# # # 12664, 15
values = np.concatenate((arr1, y_pred_three), axis=1)
lut = np.concatenate((keys, values), axis=1)


f = open('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/submission/submission.csv', 'w', newline='')
writer = csv.DictWriter(f, delimiter=',', fieldnames=header)
writer.writeheader()

new_Writer = csv.writer(f, delimiter=',')
for i in lut:
    new_Writer.writerow(['{:.3f}'.format(x) for x in i])

with ZipFile('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/submission.zip', 'w') as zip:
    zip.write('submission/submission.csv')

