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

labels_arr = np.asarray(labels.loc[:, 'LABEL_BaseExcess':'LABEL_EtCO2'])

# five for now, can be tunable depends on computer's speed
down_size = 5
rand_indx = helper.random_gen(0, 11, down_size)

X_train_tsk1 = list()
X_test_tsk1 = list()

X_train_tsk1 = np.asarray(helper.transform_list(rand_indx, train_chunks, X_train_tsk1))
X_test_tsk1 = np.asarray(helper.transform_list(rand_indx, test_chunks, X_test_tsk1))

# averaged out data
X_train_mean = X_train_tsk1.mean(axis=1)
X_test_mean = X_test_tsk1.mean(axis=1)

### X_train_tsk1 = X_train_tsk1.reshape(-1, X_train_tsk1.shape[-1])

# 18955 * 10
med_test_svm = SVC(kernel='sigmoid')

for j in range(labels_arr.shape[1]):
    med_test_svm.fit(X_train_mean, labels_arr[:, j])
    labels_pred = med_test_svm.predict(X_test_mean)

print(labels_pred[:,1])