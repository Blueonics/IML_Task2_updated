import numpy as np
import random
from random import seed
from sklearn.model_selection import RepeatedKFold
from tensorflow.keras import Sequential
from keras import layers

def make_chunks(data):
    chunks = dict()
    IDs = np.unique(data[:, 0])
    for ID in IDs:
        sel = data[:, 0] == ID
        chunks[ID] = data[sel, :]
    return chunks


def random_gen(start, stop, n_rows):
    seed(1)
    rand_list = random.sample(range(start, stop), n_rows)
    return rand_list


def transform_list(dictionary, a_list):
    for key in dictionary.keys():
        curr_rows = dictionary.get(key)
        # reduce_rows size: 5x37 and drop ID column
        a_list.append(curr_rows[:, 1:])
    return a_list


def transform_rand_list(rand_indx, dictionary, a_list):
    for key in dictionary.keys():
        curr_rows = dictionary.get(key)
        # reduce_rows size: 5x37 and drop ID column
        reduce_rows = curr_rows[rand_indx, 1:]
        a_list.append(reduce_rows)
    return a_list


def get_nn(num_in, num_out):
    NN = Sequential()
    NN.add(layers.Dense(16, input_dim=num_in, kernel_initializer='he_uniform', activation='relu'))
    NN.add(layers.Dense(num_out, activation='sigmoid'))
    NN.compile(loss='binary_crossentropy', optimizer='adam')
    return NN

