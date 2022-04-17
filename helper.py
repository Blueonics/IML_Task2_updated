import numpy as np
from keras import layers
from keras import Sequential


def imputer(df_arr, df):
    df_avg = df.mean()
    for p_idx in range(df_arr.shape[0]):
        for f_idx in range(df_arr.shape[2]):
            feature_cols = df_arr[p_idx, :, f_idx]
            nan_flag = np.where(np.isnan(feature_cols))[0]
            all_nan = (nan_flag.shape[0] == 12)
            no_nan = (nan_flag.shape[0] == 0)
            if all_nan:
                df_arr[p_idx, :, f_idx] = df_avg[f_idx]
            elif no_nan:
                df_arr[p_idx, :, f_idx] = feature_cols
            else:
                nan_avg = np.mean(feature_cols[np.where(np.isnan(feature_cols)==False)])
                for index in nan_flag:
                    df_arr[p_idx, index, f_idx] = nan_avg
    return df_arr


def batch_norm(data, mean = None, std = None):
        if mean is not None and std is not None:
            return (data - mean) / std
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / std, mean, std


def get_nn(num_in, num_out):
    NN = Sequential()
    NN.add(layers.Dense(16, input_dim=num_in, kernel_initializer='he_uniform', activation='relu'))
    NN.add(layers.Dense(num_out, activation='sigmoid'))
    NN.compile(loss='binary_crossentropy', optimizer='adam')
    return NN


