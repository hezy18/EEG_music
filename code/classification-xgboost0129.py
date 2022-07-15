import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn import preprocessing
label = np.load('label.npy')
data= np.load('data.npy')
data_withpsd = np.load('data_withpsd.npy')
print('n_cases=%d, n_features of baselin=%d, n_features of withpsd=%d'%(data.shape[0],data.shape[1],data_withpsd.shape[1]))

min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data)
min_max_scaler = preprocessing.MinMaxScaler()
data_withpsd = min_max_scaler.fit_transform(data_withpsd)

from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from scipy.stats import ttest_rel

import xgboost as xgb
from xgboost import plot_importance
import numpy as np
import warnings

seeds = [1,2,3,4,5]
model_random_state=99

# split data
N_split = 5
N_times=5
train_all_index=[]
test_all_index=[]
for time in range(N_times):
    train_all_index.append([])
    test_all_index.append([])
    kf = KFold(n_splits=N_split, shuffle=True, random_state=seeds[time])
    for train_index, test_index in  kf.split(data):
        train_all_index[time].append(train_index)
        test_all_index[time].append(test_index)

def XGB(lr=0.01,es=100,metric='error',depth=3,weight=1,gamma=0,subsample=1,col=1,alph=0,labd=0):#lr,es,metric,depth,weight,gamma,subsample,col,alph,labd
    #print('\nstart xgboost ......')
    def train_baseline():
        tmp_acc = []
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = label[train_index], label[test_index]
                
                model = xgb.XGBClassifier(booster = 'gbtree',learning_rate = lr,n_estimators = es,eval_metric=metric,
                                        max_depth = depth, min_child_weight=weight, gamma = gamma,
                                        subsample = subsample, colsample_bytree = col, reg_alpha = alph, reg_lambda = labd,
                                        use_label_encoder=False,seed = model_random_state)#lr,es,metric,depth,weight,gamma,subsample,col,alph,labd
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred)           
                tmp_acc.append(acc)

        print('acc=',np.mean(tmp_acc),'(var=',np.var(tmp_acc),')')
        return tmp_acc

    def train_withpsd():
        tmp_acc = []
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test = data_withpsd[train_index], data_withpsd[test_index]
                y_train, y_test = label[train_index], label[test_index]
                
                model = xgb.XGBClassifier(booster = 'gbtree',learning_rate = lr,n_estimators = es,eval_metric=metric,
                                        max_depth = depth, min_child_weight=weight, gamma = gamma,
                                        subsample = subsample, colsample_bytree = col, reg_alpha = alph, reg_lambda = labd,
                                        use_label_encoder=False,seed = model_random_state)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred)           
                tmp_acc.append(acc)

        print('acc=',np.mean(tmp_acc),'(var=',np.var(tmp_acc),')')
        return tmp_acc

    baselines = train_baseline()
    withpsds = train_withpsd()
    return baselines,withpsds


lr_list=[0.01,0.005]
es_list=np.arange(150,310,10)
metric_list=['merror']
depth_list= [3]
weight_list=[3, 4, 5]
gamma_list=[0.4,0.7, 1]

subsample_list=[0.8,1]
colsample_list=[0.6,0.8,1]
#alpha_list = [0,0.01,0.1,1]
#lambda_list = [0,0.1,0.5,1]


best_avg_bas=0
best_avg_psd=0
best_all_bas=[]
best_all_psd=[]
for es in es_list:
    for weight in weight_list:
        for gamma in gamma_list:
            for subsample in subsample_list:
                for colsample in colsample_list:
                    baseline_acc,withpsd_acc = XGB(0.01,es,'merror',3,weight,gamma,subsample,colsample,0,0)
                    print('lr=',0.01,'es='+str(es)+' metric='+'merror'+' depth='+str(3)+' weight='+str(weight)+
                                                    ' gamma=',gamma,'subsample=',subsample, 'colsample=',colsample,'alpha=',0,'lambda=',0)
                    if best_avg_bas < np.mean(baseline_acc):
                        best_avg_bas = np.mean(baseline_acc)
                        best_all_bas = baseline_acc
                        best_para_bas ={'lr':0.01,'es':es,'metric':'merror','depth':3,'weight':weight,
                                                                    'gamma':gamma,'subsample':subsample,'colsample':colsample,'alpha':0,'lambda':0}
                    if best_avg_psd < np.mean(withpsd_acc):
                        best_avg_psd = np.mean(withpsd_acc)
                        best_all_psd = withpsd_acc
                        best_para_psd ={'lr':0.01,'es':es,'metric':'merror','depth':3,'weight':weight,
                                                                    'gamma':gamma,'subsample':subsample,'colsample':colsample,'alpha':0,'lambda':0}
                                                    

                                                
print(best_para_bas)
print(best_avg_bas)
print(best_para_psd)
print(best_avg_psd)

results={'xgb_baseline':best_all_bas, 'xgb_withpsd':best_all_psd}
import json
with open('result.json', 'w') as f:
  json.dump(results, f)