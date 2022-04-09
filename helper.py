import numpy as np
import pandas as pd
import random
from random import seed


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


def transform_list(rand_indx, dictionary, a_list):
    for key in dictionary.keys():
        curr_rows = dictionary.get(key)
        # reduce_rows size: 5x37 and drop ID column
        reduce_rows = curr_rows[rand_indx, 1:]
        a_list.append(reduce_rows)
    return a_list

