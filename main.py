# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# subtask 1
# Data pre-processing
import sklearn as sklearn

df = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/train_features.csv')
labels = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/train_labels.csv')
ind = ['BaseExcess', 'Fibrinogen', 'AST', 'Alkalinephos', 'Bilirubin_total', 'Lactate', 'TroponinI', 'SaO2',
       'Bilirubin_direct', 'EtCO2']

X_train_tsk1 = df.loc[:, ind]
X_test_tsk1 = df.loc[:, ind]

# plt.scatter(X_test_tsk1.loc[:, 'BaseExcess'], X_train_tsk1.loc[:, 'Fibrinogen'], c='r', )
# plt.show()

labels = np.asarray(labels.loc[:, 'LABEL_BaseExcess':'LABEL_EtCO2'])

# interpolating missing values
total_features = X_train_tsk1.shape[1]

# Imputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X_train_tsk1)
X_train_imp = imp.transform(X_train_tsk1)

# per medical test, we have a rows * 12array
split_size = 12
# new_X_train = np.ones((total_features, int(X_train_imp.shape[0]/split_size), split_size))
new_X_train = np.asarray(np.array_split(X_train_imp, split_size, axis=0))
new_X_test = np.asarray(np.array_split(X_test_tsk1, split_size, axis=0))


# swap axes
new_X_train = np.swapaxes(new_X_train, 0, 2)
new_X_test = np.swapaxes(new_X_test, 0, 2)

y_pred_tsk1 = np.ones((X_test_tsk1.shape[0]/split_size, total_features))

for i in range(total_features):
    med_test_svm = SVC(kernel='sigmoid')
    med_test_svm.fit(new_X_train[i, :, :], labels[:, i])
    y_pred_tsk1[:, i] = med_test_svm.predict(new_X_test)


# if __name__ == '__main__':

# apply RBF
