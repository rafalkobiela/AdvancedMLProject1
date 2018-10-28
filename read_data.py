import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('train.txt', sep=' ')

number_of_nulls = df.isna().sum() / df.shape[0]

np.unique(np.round(number_of_nulls, 2), return_counts=True)

plt.hist(number_of_nulls, bins=100)
plt.show()

criteria = number_of_nulls < 0.3

df = df[criteria.index[criteria]]

df.fillna(df.mean(), inplace=True)

df.isna().sum() / df.shape[0]

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

newdf = df.select_dtypes(include=numerics)

X = newdf.drop('class', axis=1)
y = newdf.loc[:, 'class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))

y_train = y_train.astype(int)
y_test = y_test.astype(int)

X_train.to_csv(os.path.join('data','X_train'), index=False, sep='|')
y_train.to_csv(os.path.join('data','y_train'), index=False, sep='|')
X_test.to_csv(os.path.join('data','X_test'), index=False, sep='|')
y_test.to_csv(os.path.join('data','y_test'), index=False, sep='|')
