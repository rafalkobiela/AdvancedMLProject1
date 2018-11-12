import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def preprocess_data(INCLUDE_CAT=True, plot=False, unique_values=30, OHE=True):
    df = pd.read_csv('http://www.ipipan.eu/~teisseyrep/TEACHING/ZMUM/DANE/Projekt1/train.txt', sep=" ")

    X_test = pd.read_csv('http://www.ipipan.eu/~teisseyrep/TEACHING/ZMUM/DANE/Projekt1/testx.txt', sep=" ")

    X_test['class'] = 1

    number_of_nulls = df.isna().sum() / df.shape[0]

    if plot:
        plt.hist(number_of_nulls, bins=100)
        plt.show()


    criteria = number_of_nulls < 0.3

    df = df[criteria.index[criteria]]
    X_test = X_test[criteria.index[criteria]]

    df.fillna(df.mean(), inplace=True)
    X_test.fillna(df.mean(), inplace=True)

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    new_df = df.select_dtypes(include=numerics)
    X_test_new = X_test.select_dtypes(include=numerics)

    X = new_df.drop('class', axis=1)
    X_test_new = X_test_new.drop('class', axis=1)
    y = new_df.loc[:, 'class']

    scaler = StandardScaler()
    scaler.fit(X)
    X = pd.DataFrame(scaler.transform(X))
    X_test_new = pd.DataFrame(scaler.transform(X_test_new))

    y = y.astype(int)

    if INCLUDE_CAT:

        df_cat = df.select_dtypes(exclude=numerics)
        X_test_cat = X_test.select_dtypes(exclude=numerics)

        unique_values_col = dict()

        for col in df_cat.columns:
            unique_values_col[col] = len(df_cat.loc[:, col].unique())

        cat_cols_to_leave = []

        for key in unique_values_col.keys():
            if unique_values_col[key] < unique_values:
                cat_cols_to_leave.append(key)

        df_cat = df_cat.loc[:, cat_cols_to_leave]
        X_test_cat = X_test_cat.loc[:, cat_cols_to_leave]

        for col in df_cat.columns:
            df_cat[col].fillna(df_cat[col].mode()[0], inplace=True)
            X_test_cat[col].fillna(df_cat[col].mode()[0], inplace=True)

        if OHE:

            for col in df_cat.columns:
                try:
                    label_encoder = LabelEncoder()
                    integer_encoded = label_encoder.fit_transform(df_cat.loc[:, col].astype('str'))
                    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
                    integer_encoded_test = label_encoder.transform(X_test_cat.loc[:, col].astype('str'))
                    integer_encoded_test = integer_encoded_test.reshape(len(integer_encoded_test), 1)

                    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
                    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
                    onehot_encoded_test = onehot_encoder.transform(integer_encoded_test)

                    tmp_df = pd.DataFrame(onehot_encoded)
                    tmp_df_test = pd.DataFrame(onehot_encoded_test)

                    X = pd.concat([X, tmp_df], axis=1)
                    X_test_new = pd.concat([X_test_new, tmp_df_test], axis=1)
                except:
                    print(col)

        else:
            X = pd.concat([X.reset_index(drop=True), df_cat.reset_index(drop=True)], axis=1, ignore_index=True)
            X_test = pd.concat([X_test_new.reset_index(drop=True), X_test_cat.reset_index(drop=True)], axis=1, ignore_index=True)

        X.columns = ['Var{}'.format(str(i)) for i in range(X.shape[1])]
        X_test_new.columns = ['Var{}'.format(str(i)) for i in range(X.shape[1])]

    # preproces test data

    return X, y, X_test_new


if __name__ == "__main__":
    print(preprocess_data(False)[0].shape)
