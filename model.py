# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 10:31:18 2022

@author: trist
"""

import pandas as pd
import numpy as np

data = pd.read_csv('avila/avila-tr.txt', sep=',')
col = np.array(data.columns)
data.columns=['intercol', 'up_margin', 'low_margin', 'exploitation', 'row_number', 'mod_ratio', 'interline', 'weight', 'peak_number', 'mod_ratio/interline', 'class']
dic = {data.columns[i] : col[i] for i in range(len(col))}
data.append(dic, ignore_index = True)

data = data[data['up_margin'] < 50]
count = data["class"].value_counts()
index = data["class"].value_counts().index

data = data.drop("mod_ratio/interline", axis=1)
X_train = data.drop("class",axis=1)
Y_train = data["class"]

test = pd.read_csv('avila/avila-ts.txt', sep=',')
col = np.array(test.columns)
test.columns=['intercol', 'up_margin', 'low_margin', 'exploitation', 'row_number', 'mod_ratio', 'interline', 'weight', 'peak_number', 'mod_ratio/interline', 'class']
dic = {test.columns[i] : col[i] for i in range(len(col))}
test.append(dic, ignore_index = True)

test = test.drop("mod_ratio/interline", axis=1)
X_test = test.drop("class",axis=1)
Y_test = test["class"]
count_test = test["class"].value_counts()
x = {index[i] : count_test[i] for i in range(len(count))}

scores_modeles = pd.DataFrame(columns=['model_name', 'score'])

from sklearn.ensemble import RandomForestClassifier

g_beRF = RandomForestClassifier(bootstrap=False, n_estimators=500)
random_forest = g_beRF
modele = random_forest.fit(X_train, Y_train)

from joblib import dump, load
dump(modele, 'model.pkl')
print("Model dumped!")

rdmfor = load('model.pkl')

model_columns = list(X_train.columns)
dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")