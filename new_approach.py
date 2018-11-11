import numpy as np
import eval
import read_data as rd
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics.scorer import make_scorer
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
import time
import pickle

start_time = time.time()

categorical = False
unique_values = 30

X, y, X_test = rd.preprocess_data(INCLUDE_CAT=categorical,
                                  plot=False,
                                  unique_values=unique_values,
                                  OHE=True)

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

gbc = GradientBoostingClassifier()

n_iter_search = 40

clf = RandomizedSearchCV(gbc,
                         grid_params_random,
                         scoring=my_prec,
                         n_iter=n_iter_search,
                         cv=5,
                         n_jobs=-1,
                         verbose=10)

print('random search')

clf.fit(X, y)

# with open('data/best_model.pkl', 'rb') as file:
#     clf = pickle.load(file)

kf = KFold(n_splits=5,
           shuffle=True)

scores = []

for train_index, test_index in kf.split(X):
    X_test = X.iloc[test_index, :]
    y_test = np.array(y).ravel()[test_index]
    prob = clf.predict_proba(X_test)
    scores.append(eval.precision_at_10(y_test, prob[:, 1]))

print('Model cv score: ', np.mean(scores))

with open('models/{}-{}-{}.pkl'.format(str(clf.estimator).split('(')[0],
                                       categorical,
                                       np.round(np.mean(scores), 7)), 'wb') as file:
    pickle.dump(clf, file)

print("--- %s seconds ---" % (time.time() - start_time))
