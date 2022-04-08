# This is a sample Python script.

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
import matplotlib.pyplot as plt


df = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/train_features.csv')
labels = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/train_labels.csv')
ind = ['BaseExcess', 'Fibrinogen', 'AST', 'Alkalinephos', 'Bilirubin_total', 'Lactate', 'TroponinI', 'SaO2',
       'Bilirubin_direct', 'EtCO2']

X_train_tsk1 = df.loc[:, ind]
X_test_tsk1 = df.loc[:, ind]

# plt.scatter(X_test_tsk1.loc[:, 'BaseExcess'], X_train_tsk1.loc[:, 'Fibrinogen'], c='r', )
# plt.show()

labels_arr = np.asarray(labels.loc[:, 'LABEL_BaseExcess':'LABEL_EtCO2'])
total_features = X_train_tsk1.shape[1]

# Imputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X_train_tsk1)

X_train_imp = imp.transform(X_train_tsk1)
X_test_imp = imp.transform(X_test_tsk1)

# per medical test, we have a rows * 12array
split_size = 12
# new_X_train = np.ones((total_features, int(X_train_imp.shape[0]/split_size), split_size))
new_X_train = np.swapaxes(np.asarray(np.array_split(X_train_imp, split_size, axis=0)), 0, 2)
new_X_test = np.swapaxes(np.asarray(np.array_split(X_test_imp, split_size, axis=0)), 0, 2)

X_train_reshape = new_X_train.reshape(-1, new_X_train.shape[-1])
labels_reshape = labels_arr.reshape(X_train_reshape.shape[0])
X_test_reshape = new_X_test.reshape(-1, new_X_test.shape[-1])
# print(X_train_reshape.shape)
# print(labels_reshape.shape)

# y_pred_tsk1 = np.ones((int(X_test_tsk1.shape[0]/split_size), total_features))
med_test_svm = SVC(kernel='sigmoid')
med_test_svm.fit(X_test_reshape, labels_reshape)
y_pred_tsk1 = med_test_svm.predict(X_test_reshape)
# print(y_pred_tsk1.shape)


# for i in range(total_features):
    # med_test_svm.fit(new_X_train[0, :, :], labels[:, 0])
    # y_pred_tsk1[:, i] = med_test_svm.predict(new_X_test[i, :, :])


# if __name__ == '__main__':

# apply RBF
