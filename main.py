import numpy as np
import pandas as pd
from subtask1 import subtask1_predict
from subtask3 import subtask3_predict
import helper

df = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/train_features.csv')
df_test = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/test_features.csv')
labels = pd.read_csv('C:/Users/Lannan Jiang/PycharmProjects/IML_Task2/train_labels.csv')

# subtask3_predict(df, df_test, labels)
# subtask1_predict(df, df_test, labels)
