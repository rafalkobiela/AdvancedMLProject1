import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
# import xgboost as xgb
from sklearn import metrics
import lightgbm as lgb
import os
import eval

X_train = pd.read_csv(os.path.join('data', 'X_train'), sep='|')
y_train = pd.read_csv(os.path.join('data', 'y_train'), sep='|', header=None)
X_test = pd.read_csv(os.path.join('data', 'X_test'), sep='|')
y_test = pd.read_csv(os.path.join('data', 'y_test'), sep='|', header=None)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Logistic Regression
clf = LogisticRegression(random_state=42, class_weight='balanced', penalty='l2')
clf.fit(X_train, y_train)

prob = clf.predict_proba(X_test)
clases = clf.predict(X_test)

print('Logistic regression :',
      eval.precision_at_10(y_test, prob[:, 1]))

# Random Forest
clf = RandomForestClassifier(random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

prob = clf.predict_proba(X_test)
clases = clf.predict(X_test)

print("Random forest:",
      eval.precision_at_10(y_test, prob[:, 1]))

# Gradient boosting

clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)

prob = clf.predict_proba(X_test)
clases = clf.predict(X_test)

print("Gradient boosting:",
      eval.precision_at_10(y_test, prob[:, 1]))


#Ada


clf = AdaBoostClassifier()
clf.fit(X_train, y_train)

prob = clf.predict_proba(X_test)
clases = clf.predict(X_test)

print("AdaBoost:",
      eval.precision_at_10(y_test, prob[:, 1]))



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
clf.fit(X_train, y_train)

prob = clf.predict_proba(X_test)
clases = clf.predict(X_test)

print("LGBM:",
      eval.precision_at_10(y_test, prob[:, 1]))
