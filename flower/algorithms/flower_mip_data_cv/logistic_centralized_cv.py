from sklearn.linear_model import LogisticRegression
from typing import Dict
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score

#model = LogisticRegression()

xvars = ['lefthippocampus', 'leftamygdala']
yvars = ['gender']

dataframes_list=[]

for i in range(10):
    curr_filename = '/Users/aglenis/MIP-Engine/tests/test_data/dementia_v_0_1/ppmi'+str(i)+'.csv'
    curr_df = pd.read_csv(curr_filename)
    dataframes_list.append(curr_df)

full_data = pd.concat(dataframes_list)

models_list = []

n_splits =5

kf = KFold(n_splits=n_splits)

for i in range(n_splits):
    models_list.append(LogisticRegression(solver='saga'))

le = preprocessing.LabelEncoder()
le.fit(['M','F'])
print(list(le.classes_))

X = full_data[xvars].values
y = le.transform(full_data[yvars].values.ravel())

i=0
for train, test in kf.split(X):
    X_train = X[train]
    y_train = y[train]

    X_test = X[test]
    y_test = y[test]

    models_list[i].fit(X_train,y_train)
    df = pd.DataFrame()
    print(models_list[i].coef_)
    y_pred = models_list[i].predict(X)
    df['y_pred'] = y_pred
    curr_accuracy_whole = accuracy_score(y,y_pred)
    print('curr_accuracy for the whole dataset '+str(curr_accuracy_whole))
    df.to_csv('y_pred_centralized_'+str(i)+'.csv',index=False)
    accuracy = models_list[i].score(X_test, y_test)
    print('accuracy of centralized logistic regression is '+str(accuracy)+' fold = '+str(i))

    i+=1

#model.fit(X_train,y_train)
