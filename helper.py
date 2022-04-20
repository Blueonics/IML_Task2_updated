import numpy as np
from keras import layers
from keras import Sequential
from sklearn.decomposition import PCA


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
                nan_avg = np.mean(feature_cols[np.where(np.isnan(feature_cols) == False)])
                for index in nan_flag:
                    df_arr[p_idx, index, f_idx] = nan_avg
    return df_arr


def pca_for_time(X_imputed, X_test_imputed, n_components):
    X_train_pca = np.ones((X_imputed.shape[0], X_imputed.shape[2], n_components))
    X_test_pca = np.ones((X_test_imputed.shape[0], X_test_imputed.shape[2], n_components))

    X_train_T = np.transpose(X_imputed, (0, 2, 1))
    X_test_T = np.transpose(X_test_imputed, (0, 2, 1))

    nn_pca = PCA(n_components=n_components)
    for i in range(X_train_T.shape[0]):
        feature_train = X_train_T[i, :, :]
        nn_pca.fit(feature_train)
        X_train_transformed = nn_pca.transform(feature_train)
        if i < X_test_imputed.shape[0]:
            feature_test = X_test_T[i, :, :]
            X_test_transformed = nn_pca.transform(feature_test)
            X_test_pca[i, :, :] = X_test_transformed
        X_train_pca[i, :, :] = X_train_transformed
    X_train_T = np.transpose(X_train_pca, (0, 1, 2))
    X_test_T = np.transpose(X_test_pca, (0, 1, 2))
    print('PCA reshaped', X_train_T.shape)
    print('PCA reshaped', X_test_T.shape)
    return X_train_T, X_test_T


def average_dim(X, step=2):
    X_T = np.transpose(X, (0, 2, 1)) #18895 x 12 x 35
    X_new_T = np.ones((X_T.shape[0], X_T.shape[1], int(X_T.shape[2]/step))) # 18995 x 35 x 6
    print(X_new_T.shape)
    for p_idx in range(X_T.shape[0]):# 18995
        for f_idx in range(X_T.shape[1]): # 35
            for i in range(0, X_T.shape[2], step): # 0, 12, step
                mean = np.mean(X_T[p_idx, f_idx, i:(i+step)])
                X_new_T[p_idx, f_idx, int(i/2)] = mean
    print("pass!!")
    X_dim_down = np.transpose(X_new_T, (0, 1, 2))
    return X_dim_down


def batch_norm(data, mean=None, std=None):
    if mean is not None and std is not None:
        return (data - mean) / std
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std, mean, std


def get_nn(num_in, num_out):
    NN = Sequential()
    NN.add(layers.Dense(32, input_dim=num_in, kernel_initializer='he_uniform', activation='relu'))
    NN.add(layers.Dense(16, kernel_initializer='he_uniform', activation='relu'))
    NN.add(layers.Dense(num_out, activation='sigmoid'))
    NN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return NN
