import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import eval
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.ensemble import GradientBoostingClassifier
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


scaler = StandardScaler()
scaler.fit(X)

X = pd.DataFrame(scaler.transform(X))
y = y.astype(int)


my_prec = make_scorer(eval.precision_at_10,
                      needs_proba=True,
                      greater_is_better=True)


grid_params = {'learning_rate': np.arange(0.01, 1, 0.1),
               'n_estimators': np.arange(10,300,100),
               'min_samples_split': np.arange(2,50,20),
               'max_depth': np.arange(2,10,5)}

gbc = GradientBoostingClassifier()

clf = GridSearchCV(gbc,
                   grid_params,
                   scoring = 'roc_auc',
                   cv=5,
                   verbose=10)

print('grid search')

clf.fit(X, y)

clf.best_params_