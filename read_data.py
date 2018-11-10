import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def preprocess_data(INCLUDE_CAT=True, plot=False, unique_values=30):
    df = pd.read_csv('train.txt', sep=' ')

    X_test = pd.read_csv('testx.txt', sep=" ")
    X_test['class'] = 1

    number_of_nulls = df.isna().sum() / df.shape[0]

    np.unique(np.round(number_of_nulls, 2), return_counts=True)

    if plot:
        plt.hist(number_of_nulls, bins=100)
        plt.show()

    criteria = number_of_nulls < 0.3

    df = df[criteria.index[criteria]]
    X_test = X_test[criteria.index[criteria]]

    df.fillna(df.mean(), inplace=True)
    X_test.fillna(df.mean(), inplace=True)

    df.isna().sum() / df.shape[0]

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    new_df = df.select_dtypes(include=numerics)
    X_test = X_test.select_dtypes(include=numerics)

    X = new_df.drop('class', axis=1)
    X_test = X_test.drop('class', axis=1)
    y = new_df.loc[:, 'class']

    scaler = StandardScaler()
    scaler.fit(X)
    X = pd.DataFrame(scaler.transform(X))
    X_test = pd.DataFrame(scaler.transform(X_test))

    y = y.astype(int)

    if INCLUDE_CAT:

        df_cat = df.select_dtypes(exclude=numerics)

        unique_values_col = dict()

        for col in df_cat.columns:
            unique_values_col[col] = len(df_cat.loc[:, col].unique())

        cat_cols_to_leave = []

        for key in unique_values_col.keys():
            if unique_values_col[key] < unique_values:
                cat_cols_to_leave.append(key)

        df_cat = df_cat.loc[:, cat_cols_to_leave]

        df_cat.fillna(lambda x: x.fillna(x.mode()[0]), inplace=True)

        for col in df_cat.columns:
            label_encoder = LabelEncoder()
            integer_encoded = label_encoder.fit_transform(df_cat.loc[:, col].astype('str'))
            onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
            tmp_df = pd.DataFrame(onehot_encoded)
            X = pd.concat([X, tmp_df], axis=1)

        X.columns = ['Var{}'.format(str(i)) for i in range(X.shape[1])]

    # preproces test data

    return X, y, X_test


if __name__ == "__main__":
    print(preprocess_data(False)[0].shape)
