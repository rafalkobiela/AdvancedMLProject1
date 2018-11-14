import numpy as np
import eval
import read_data as rd
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics.scorer import make_scorer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import randint as sp_randint
import lightgbm as lgb
from scipy.stats import uniform
import time
import pickle

start_time = time.time()

categorical = True
unique_values = 20

try:
    X = pd.read_csv("data/X.csv", sep='|')
    y = pd.read_csv("data/y.csv", sep='|')
    X = pd.read_csv("data/X_test.csv", sep='|')
except:
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

grid_params = {'learning_rate': np.arange(0.01, 1, 0.1),
               'n_estimators': np.arange(10, 300, 100),
               'min_samples_split': np.arange(2, 50, 20),
               'max_depth': np.arange(2, 10, 5)}

grid_params_random = {'learning_rate': uniform(0.1, 1),
                      'n_estimators': sp_randint(100, 300),
                      'min_samples_split': sp_randint(2, 20),
                      'max_depth': sp_randint(2, 10)}

params = {
    'learning_rate': uniform(0.1, 1),
    'num_leaves': sp_randint(20, 200),
    'max_depth': sp_randint(2, 20),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1),
    'min_child_weight': uniform(0, 0.1),
    'min_child_samples': sp_randint(10, 100),
    'feature_fraction': uniform(0.3, 0.7),
    'bagging_fraction': uniform(0.3, 0.7)
}

gbc = GradientBoostingClassifier()

# gbc = lgb.LGBMClassifier(is_unbalance=True, objective='binary', n_jobs=1)

n_iter_search = 20

clf = RandomizedSearchCV(gbc,
                         grid_params_random,
                         scoring=my_prec,
                         n_iter=n_iter_search,
                         cv=5,
                         n_jobs=-1,
                         verbose=10)

print('random search')

clf.fit(X, y)

gbc = lgb.LGBMClassifier(is_unbalance=True, objective='binary', **clf.best_params_)

scores = cross_val_score(clf, X, y, cv=5, scoring=my_prec)

print('Model cv score: ', np.mean(scores))

with open('models/{}-{}-{}.pkl'.format(str(clf.estimator).split('(')[0],
                                       categorical,
                                       np.round(np.mean(scores), 7)), 'wb') as file:
    pickle.dump(clf, file)

print("--- %s seconds ---" % (time.time() - start_time))
