from sklearn.linear_model import LogisticRegression
from typing import Dict
import pandas as pd
import numpy as np
from sklearn import preprocessing

model = LogisticRegression()

xvars = ['lefthippocampus', 'leftamygdala']
yvars = ['gender']

dataframes_list=[]

for i in range(10):
    curr_filename = '/Users/aglenis/MIP-Engine/tests/test_data/dementia_v_0_1/ppmi'+str(i)+'.csv'
    curr_df = pd.read_csv(curr_filename)
    dataframes_list.append(curr_df)

full_data = pd.concat(dataframes_list)


le = preprocessing.LabelEncoder()
le.fit(['M','F'])
print(list(le.classes_))

X_test = full_data[xvars].values
y_test = le.transform(full_data[yvars].values.ravel())
X_train = X_test
y_train = y_test

model.fit(X_train,y_train)

accuracy = model.score(X_test, y_test)

print('accuracy of centralized logistic regression is '+str(accuracy))
