import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from subtask1 import subtask1_predict
from subtask2 import subtask2_predict
from subtask3 import subtask3_predict
import helper
from zipfile import ZipFile
import csv

df = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/train_features.csv')
df_test = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/test_features.csv')
labels = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/train_labels.csv')


PID = np.asarray(df_test.loc[:, 'pid'].unique())
keys = PID.reshape((PID.shape[0], 1))

#
labels = labels.drop(['pid'], axis=1)
#
labels_one = subtask1_predict(df, df_test, labels)
# # # labels_two = subtask2_predict(df, df_test, labels)
# # # y_pred_three = subtask3_predict(df, df_test, labels)
# #

# labels_one = np.ones((12664, 10))
labels_two = np.ones((12664, 1))
y_pred_three = np.ones((12664, 4))


arr1 = np.concatenate((labels_one, labels_two), axis=1)
# # # 12664, 15
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
#
