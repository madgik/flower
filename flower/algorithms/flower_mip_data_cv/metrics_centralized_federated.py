from sklearn.linear_model import LogisticRegression
from typing import Dict
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

#model = LogisticRegression()

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

X = full_data[xvars].values
y = le.transform(full_data[yvars].values.ravel())

for i in range(5):
    curr_filename = 'y_pred_centralized_'+str(i)+'.csv'
    curr_df = pd.read_csv(curr_filename)
    curr_y_pred = curr_df['y_pred'].values
    accuracy = accuracy_score(y,curr_y_pred)
    f1_score1 = f1_score(y, curr_y_pred, average='macro')
    roc_score = roc_auc_score(y, curr_y_pred,average='macro')
    print('accuracy for model '+str(i)+' is '+str(accuracy))
    print('f1 score for model '+str(i)+' is '+str(f1_score1))
    print('roc score for model '+str(i)+' is '+str(roc_score))

for i in range(5):
    curr_filename = 'y_pred_'+str(i)+'_federated.csv'
    curr_df = pd.read_csv(curr_filename)
    curr_y_pred = curr_df['y_pred'].values
    accuracy = accuracy_score(y,curr_y_pred)
    f1_score1 = f1_score(y, curr_y_pred, average='macro')
    roc_score = roc_auc_score(y, curr_y_pred,average='macro')
    print('accuracy for federated model '+str(i)+' is '+str(accuracy))
    print('f1 score for federated model '+str(i)+' is '+str(f1_score1))
    print('roc score for federated model '+str(i)+' is '+str(roc_score))
