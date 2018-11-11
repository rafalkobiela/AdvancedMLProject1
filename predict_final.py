import numpy as np
import pandas as pd
import read_data as rd
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os

files = os.listdir('models')

files_dict = {i: float(i.split('-')[-1].replace('.pkl', '')) for i in files}

max_key_score = None
for key in files_dict.keys():
    if max_key_score is None:
        max_key_score = key
        tmp = files_dict[key]
    elif tmp < files_dict[key]:
        max_key_score = key
        tmp = files_dict[key]

max_key_score = 'GradientBoostingClassifier-True-0.5233129.pkl'

with open('models/{}'.format(max_key_score), 'rb') as file:
    model = pickle.load(file)

include_cat = max_key_score.split('-')[1] == 'True'

X, y, X_test = rd.preprocess_data(include_cat, False, 20, True)

# X.to_csv("X_toR.csv", index=False, sep='|')
# y.to_csv("y_toR.csv", index=False, sep='|')

clf = GradientBoostingClassifier(**model.best_params_)

clf.fit(X, y)

prob = pd.DataFrame({"RAFKOB": clf.predict_proba(X_test)[:, 1]})

prob.to_csv("RAFKOB.txt", index=False)
