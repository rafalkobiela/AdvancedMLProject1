import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
# import xgboost as xgb
from sklearn import metrics
import lightgbm as lgb
import os
import eval

# X_train = pd.read_csv(os.path.join('data', 'X_train'), sep='|')
# y_train = pd.read_csv(os.path.join('data', 'y_train'), sep='|', header=None)
# X_test = pd.read_csv(os.path.join('data', 'X_test'), sep='|')
# y_test = pd.read_csv(os.path.join('data', 'y_test'), sep='|', header=None)

# y_train = y_train.values.ravel()
# y_test = y_test.values.ravel()

try:

    X = pd.read_csv("data/X.csv", sep='|')
    y = pd.read_csv("data/y.csv", sep='|', header=None)
    X_test = pd.read_csv("data/X_test.csv", sep='|')
    print("datasets read from disk")
    print('X: ', X.shape)
    print('y: ', y.shape)
    print('X_test: ', X_test.shape)
    y = y.astype(int)
except Exception as e:
    print(e)
    print("Create dataset")
    X, y, X_test = rd.preprocess_data(INCLUDE_CAT=categorical,
                                      plot=False,
                                      unique_values=unique_values,
                                      OHE=True)

    X.to_csv('data/X.csv', index=False, sep='|')
    y.to_csv('data/y.csv', index=False, sep='|')
    X_test.to_csv('data/X_test.csv', index=False, sep='|')
    print("dataset created")

my_prec = make_scorer(eval.precision_at_10,
                      needs_proba=True,
                      greater_is_better=True)

# Logistic Regression
clf = LogisticRegression(random_state=42, penalty='l2')

scores = cross_val_score(clf, X, y, cv=5, scoring=my_prec)

print('Logistic regression:', np.mean(scores))

# Random Forest
clf = RandomForestClassifier(random_state=42, class_weight='balanced')
scores = cross_val_score(clf, X, y, cv=5, scoring=my_prec)

print("Random forest:", np.mean(scores))

# Gradient boosting

clf = GradientBoostingClassifier()
scores = cross_val_score(clf, X, y, cv=5, scoring=my_prec)

print("Gradient boosting:", np.mean(scores))

#Ada


clf = AdaBoostClassifier()

scores = cross_val_score(clf, X, y, cv=5, scoring=my_prec)

print("AdaBoost:", np.mean(scores))




# XGBoost

# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test, label=y_test)

# param = {'max_depth': 3,
#          'eta': 0.9,
#          'silent': 1,
#          'objective': 'binary:logistic',
#          'nthread': 4,
#          'eval_metric': 'auc'}

# evallist = [(dtest, 'eval'), (dtrain, 'train')]

# num_round = 10

# bst = xgb.train(param, dtrain, 10, evallist, verbose_eval=0)

# prob = bst.predict(dtest)

# print("XGBoost:",
#       eval.precision_at_10(y_test, prob))


# LGBM

params = {
    'objective' :'binary',
    'learning_rate' : 0.02,
    'num_leaves' : 76,
    'feature_fraction': 0.64, 
    'bagging_fraction': 0.8, 
    'bagging_freq':1,
    'boosting_type' : 'gbdt',
    'metric': 'binary_logloss',
    'is_unbalance' : True,
    'n_jobs' : 3
}



clf = lgb.LGBMClassifier(
   **params
)

scores = cross_val_score(clf, X, y, cv=5, scoring=my_prec)

print("LGBM:", np.mean(scores))
