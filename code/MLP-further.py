import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn import preprocessing
rating = np.load('rating.npy')
label = np.load('label.npy')
data= np.load('data.npy')
data_withpsd = np.load('data_withpsd.npy')
print('n_cases=%d, n_features of baselin=%d, n_features of withpsd=%d'%(data.shape[0],data.shape[1],data_withpsd.shape[1]))

min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data)
min_max_scaler = preprocessing.MinMaxScaler()
data_withpsd = min_max_scaler.fit_transform(data_withpsd)

from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor, MLPClassifier
from scipy.stats import ttest_rel
import json

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

def MLP_coef():
    print('\nstart MLP ......')
    def train_baseline(hidden, activate, solve, l2):
        coef_dict={}
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = rating[train_index], rating[test_index]
                
                clf = MLPRegressor(hidden_layer_sizes = hidden, activation=activate, solver = solve, alpha=l2, max_iter = 1000,random_state=model_random_state)
            
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            
                mse = mean_squared_error(y_test, y_pred)           
                coef=[]
                for i in range(len(clf.coefs_)):
                    coef.append(clf.coefs_[i].tolist())
                coef_dict[mse] = coef
        return coef_dict
    
    def train_withpsd(hidden, activate, solve, l2):
        coef_dict={}
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test = data_withpsd[train_index], data_withpsd[test_index]
                y_train, y_test = rating[train_index], rating[test_index]
                
                clf = MLPRegressor(hidden_layer_sizes = hidden, activation=activate, solver = solve, alpha=l2, max_iter = 1000,random_state=model_random_state)
            
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            
                mse = mean_squared_error(y_test, y_pred)           
                
                coef=[]
                for i in range(len(clf.coefs_)):
                    coef.append(clf.coefs_[i].tolist())
                coef_dict[mse] = coef
        return coef_dict

    print('\tFor Baseline:')
    #hidden_list=[(100,),(50,),(30,),(10,),(10,10),(30,10),(10,30),(30,30),(25,10,5)]
    #activate_list = activate_list = ['identity','logistic','tanh', 'relu']
    #solve_list = ['lbfgs', 'adam','sgd']
    #l2_list=[1e-6,1e-4,1e-2,0]
    baseline_dict = train_baseline((30,10), 'tanh', 'sgd', 0.01)
    with open('baseline_coef.json', 'w') as f:
        json.dump(baseline_dict, f)


    print('\tFor Withpsd:')
    withpsd_dict =  train_withpsd((30, 30), 'identity', 'sgd', 0.01)
    with open('withpsd_coef.json', 'w') as f:
        json.dump(withpsd_dict, f)
                
#MLP_coef()

def ablation():
    def train_withpsd(hidden, activate, solve, l2, delete):
        if delete==0:
            data_withpsd_del = data_withpsd
        else:
            data_withpsd_del = np.delete(data_withpsd, delete, axis=1)
        tmp_acc = []
        print(data_withpsd_del.shape)
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]

                X_train, X_test = data_withpsd_del[train_index], data_withpsd_del[test_index]
                y_train, y_test = rating[train_index], rating[test_index]
                
                clf = MLPRegressor(hidden_layer_sizes = hidden, activation=activate, solver = solve, alpha=l2, max_iter = 1000,random_state=model_random_state)
            
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            
                mse = mean_squared_error(y_test, y_pred)           
                tmp_acc.append(mse)
        print('delete='+str(delete),'mse=',np.mean(tmp_acc))
        return tmp_acc

    abla_dict= {}
    abla_dict[0] = train_withpsd((30, 30), 'identity', 'sgd', 0.01, 0)
    delete_list = [[0,5,10,15,20],[1,6,11,16,21],[2,7,12,17,22],[3,8,13,18,23],[4,9,14,19,24]]   
    band = ['4-8','8-12','12-30','30-45','all']
    for i in range(5):
        abla_dict[band[i]] = train_withpsd((30, 30), 'identity', 'sgd', 0.01, delete_list[i])
    with open('abla_band.json', 'w') as f:
        json.dump(abla_dict, f)

    abla_dict= {}
    abla_dict[0] = train_withpsd((30, 30), 'identity', 'sgd', 0.01, 0)
    delete_list = [[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19],[20,21,22,23,24]]
    for i in range(5):
        abla_dict[band[i]] = train_withpsd((30, 30), 'identity', 'sgd', 0.01, delete_list[i])
    with open('abla_elec.json', 'w') as f:
        json.dump(abla_dict, f)
ablation()

def Clf_MLP_coef():
    print('\nstart MLP ......')
    def train_baseline(hidden, activate, solve, l2):
        coef_dict={}
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = label[train_index], label[test_index]

                clf = MLPClassifier(hidden_layer_sizes = hidden, activation=activate, solver = solve, alpha=l2, max_iter = 1000,random_state=model_random_state)
            
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            
                acc = accuracy_score(y_test, y_pred)             
                coef=[]
                for i in range(len(clf.coefs_)):
                    coef.append(clf.coefs_[i].tolist())
                coef_dict[acc] = coef
        return coef_dict
    
    def train_withpsd(hidden, activate, solve, l2):
        coef_dict={}
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test = data_withpsd[train_index], data_withpsd[test_index]
                y_train, y_test = label[train_index], label[test_index]

                clf = MLPClassifier(hidden_layer_sizes = hidden, activation=activate, solver = solve, alpha=l2, max_iter = 1000,random_state=model_random_state)
            
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            
                acc = accuracy_score(y_test, y_pred)             
                coef=[]
                for i in range(len(clf.coefs_)):
                    coef.append(clf.coefs_[i].tolist())
                coef_dict[acc] = coef

        return coef_dict

    print('\tFor Baseline:')
    #hidden_list=[(100,),(50,),(30,),(10,),(10,10),(30,10),(10,30),(30,30),(25,10,5)]
    #activate_list = activate_list = ['identity','logistic','tanh', 'relu']
    #solve_list = ['lbfgs', 'adam','sgd']
    #l2_list=[1e-6,1e-4,1e-2,0]
    baseline_dict = train_baseline((100,), 'tanh', 'adam', 1e-06)
    with open('clf_baseline_coef.json', 'w') as f:
        json.dump(baseline_dict, f)


    print('\tFor Withpsd:')
    withpsd_dict =  train_withpsd((10, 30), 'tanh', 'adam', 0.01)
    with open('clf_withpsd_coef.json', 'w') as f:
        json.dump(withpsd_dict, f)
                
#Clf_MLP_coef()

def Clf_ablation():
    def train_withpsd(hidden, activate, solve, l2, delete):
        if delete==0:
            data_withpsd_del = data_withpsd
        else:
            data_withpsd_del = np.delete(data_withpsd, delete, axis=1)
        print(data_withpsd_del.shape)
        tmp_acc = []
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test = data_withpsd_del[train_index], data_withpsd_del[test_index]
                y_train, y_test = label[train_index], label[test_index]

                clf = MLPClassifier(hidden_layer_sizes = hidden, activation=activate, solver = solve, alpha=l2, max_iter = 1000,random_state=model_random_state)
            
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            
                acc = accuracy_score(y_test, y_pred)       
                tmp_acc.append(acc)
        print('delete='+str(delete),'acc=',np.mean(tmp_acc))
        return tmp_acc

    abla_dict= {}
    abla_dict[0] = train_withpsd((10, 30), 'tanh', 'adam', 0.01, 0)
    delete_list = [[0,5,10,15,20],[1,6,11,16,21],[2,7,12,17,22],[3,8,13,18,23],[4,9,14,19,24]]
    band = ['4-8','8-12','12-30','30-45','all']
    for i in range(5):
        abla_dict[band[i]] = train_withpsd((10, 30), 'tanh', 'adam', 0.01, delete_list[i])
    with open('clf_abla_band.json', 'w') as f:
        json.dump(abla_dict, f)
    
    abla_dict= {}
    abla_dict[0] = train_withpsd((10, 30), 'tanh', 'adam', 0.01, 0)
    delete_list = [[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19],[20,21,22,23,24]]
    for i in range(5):
        abla_dict[band[i]] = train_withpsd((10, 30), 'tanh', 'adam', 0.01, delete_list[i])
    with open('clf_abla_elec.json', 'w') as f:
        json.dump(abla_dict, f)

Clf_ablation()

