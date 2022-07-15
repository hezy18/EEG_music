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

from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from scipy.stats import ttest_rel

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

def SVM():
    
    print('\nstart SVM ......')
    def train_baseline(c, kernel, gamma=1/data.shape[1]):
        tmp_acc = []
        
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = label[train_index], label[test_index]
                
                clf = SVC(C=c, kernel=kernel, gamma=gamma, decision_function_shape='ovo',random_state=model_random_state)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            
                acc = accuracy_score(y_test, y_pred)           
                tmp_acc.append(acc)
                
        print('C='+str(c)+',kernel='+kernel+',gamma'+str(gamma)+
                ':\tacc=',np.mean(tmp_acc),'(var=',np.var(tmp_acc),')')
        return tmp_acc
    
    def train_withpsd(c, kernel, gamma=1/data_withpsd.shape[1]):
        tmp_acc = []
    
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test = data_withpsd[train_index], data_withpsd[test_index]
                y_train, y_test = label[train_index], label[test_index]

                clf = SVC(C=c, kernel=kernel, gamma=gamma, decision_function_shape='ovo',random_state=model_random_state)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            
                acc = accuracy_score(y_test, y_pred)           
                tmp_acc.append(acc)

        print('C='+str(c)+',kernel='+kernel+',gamma'+str(gamma)+
                ':\tacc=',np.mean(tmp_acc),'(var=',np.var(tmp_acc),')')
        
        return tmp_acc
    
    print('\tFor Baseline:')
    baseline_acc =[]
    baseline_best_acc = 0
    baseline_param={}
    C_list = [0.1,1,10,100,1000]
    kernel_list = ['linear','poly','rbf', 'sigmoid']
    gamma_list = [1/100,1/50,1/10,1,5]
    for kernel in kernel_list:
        for c in C_list:
            if kernel == 'linear':
                tmp_acc = train_baseline(c,kernel)
                if np.mean(tmp_acc)>baseline_best_acc:
                    baseline_best_acc = np.mean(tmp_acc)
                    baseline_acc=tmp_acc
                    baseline_param = {'c':c,'kernel':kernel,'gamma':'0'}
            else:
                for gamma in gamma_list:
                    tmp_acc=train_baseline(c,kernel,gamma)
                    if np.mean(tmp_acc)>baseline_best_acc:
                        baseline_best_acc = np.mean(tmp_acc)
                        baseline_acc=tmp_acc
                        baseline_param = {'c':c,'kernel':kernel,'gamma':gamma}
            
    print('best_acc=',np.mean(baseline_acc),'(var=',np.var(baseline_acc),')')
    
    print('\tFor Withpsd:')
    withpsd_acc =[]
    withpsd_best_acc = 0
    withpsd_param={}
    for kernel in kernel_list:
        for c in C_list:
            if kernel == 'linear':
                tmp_acc = train_withpsd(c,kernel)
                if np.mean(tmp_acc)>withpsd_best_acc:
                    withpsd_best_acc = np.mean(tmp_acc)
                    withpsd_acc=tmp_acc
                    withpsd_param = {'c':c,'kernel':kernel,'gamma':'0'}
            else:
                for gamma in gamma_list:
                    tmp_acc = train_withpsd(c,kernel,gamma)
                    if np.mean(tmp_acc)>withpsd_best_acc:
                        withpsd_best_acc = np.mean(tmp_acc)
                        withpsd_acc=tmp_acc
                        withpsd_param = {'c':c,'kernel':kernel,'gamma':gamma}
                        
    print('best_acc=',np.mean(withpsd_acc),'(var=',np.var(withpsd_acc),')\n')
    print('baseline=',np.mean(baseline_acc),'withpsd=',np.mean(withpsd_acc),(np.mean(withpsd_acc)-np.mean(baseline_acc))/np.mean(baseline_acc))
    print(ttest_rel(baseline_acc,withpsd_acc))
    baseline={'acc':baseline_acc,'param':baseline_param}
    withpsd={'acc':withpsd_acc,'param':withpsd_param}
    return baseline,withpsd

def GBDT():
    print('\nstart GBDT ......')
    def train_baseline(lr,es):
        tmp_acc = []
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = label[train_index], label[test_index]
                
                clf = GradientBoostingClassifier(learning_rate = lr, n_estimators =es, random_state=model_random_state)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            
                acc = accuracy_score(y_test, y_pred)           
                tmp_acc.append(acc)

        print('learning_rate=',lr,'n_estimators',es,':\tacc=',np.mean(tmp_acc),'(var=',np.var(tmp_acc),')')
        return tmp_acc
    
    def train_withpsd(lr,es):
        tmp_acc = []
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test = data_withpsd[train_index], data_withpsd[test_index]
                y_train, y_test = label[train_index], label[test_index]
                
                clf = GradientBoostingClassifier(learning_rate = lr, n_estimators =es,random_state=model_random_state)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            
                acc = accuracy_score(y_test, y_pred)          
                tmp_acc.append(acc)

        print('learning_rate=',lr,'n_estimators',es,':\tacc=',np.mean(tmp_acc),'(var=',np.var(tmp_acc),')')
        return tmp_acc
    
    print('\tFor Baseline:')
    baseline_acc =[]
    baseline_best_acc = 0
    baseline_param={}
    lr_list=[0.0001,0.001,0.01]
    es_list=list(np.arange(100,400,10))
    for lr in lr_list:
        for es in es_list:
            tmp_acc = train_baseline(lr,es)
            if np.mean(tmp_acc)>baseline_best_acc:
                baseline_best_acc = np.mean(tmp_acc)
                baseline_acc=tmp_acc
                baseline_param={'lr':str(lr),'es':str(es)}
    print('best_acc=',np.mean(baseline_acc),'(var=',np.var(baseline_acc),')\n')

    print('\tFor Withpsd:')
    withpsd_acc =[]
    withpsd_best_acc = 0
    withpsd_param={}
    for lr in lr_list:
        for es in es_list:
            tmp_acc = train_withpsd(lr,es)
            if np.mean(tmp_acc)>withpsd_best_acc:
                withpsd_best_acc = np.mean(tmp_acc)
                withpsd_acc=tmp_acc
                withpsd_param={'lr':str(lr),'es':str(es)}
    print('best_acc=',np.mean(withpsd_acc),'(var=',np.var(withpsd_acc),')\n')

    print(ttest_rel(baseline_acc,withpsd_acc))
    baseline={'acc':baseline_acc,'param':baseline_param}
    withpsd={'acc':withpsd_acc,'param':withpsd_param}
    return baseline,withpsd

def RF():
    print('\nstart RF ......')
    def train_baseline(es):
        tmp_acc = []
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = label[train_index], label[test_index]
                
                clf = RandomForestClassifier(n_estimators =es,oob_score=True,random_state=model_random_state)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            
                acc = accuracy_score(y_test, y_pred)           
                tmp_acc.append(acc)

        print('n_estimators',es,':\tacc=',np.mean(tmp_acc),'(var=',np.var(tmp_acc),')')
        return tmp_acc
    
    def train_withpsd(es):
        tmp_acc = []
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test =data_withpsd[train_index], data_withpsd[test_index]
                y_train, y_test = label[train_index], label[test_index]
                
                clf = RandomForestClassifier(n_estimators =es,oob_score=True,random_state=model_random_state)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            
                acc = accuracy_score(y_test, y_pred)           
                tmp_acc.append(acc)

        print('n_estimators',es,':\tacc=',np.mean(tmp_acc),'(var=',np.var(tmp_acc),')')
        return tmp_acc

    print('\tFor Baseline:')
    baseline_acc =[]
    baseline_best_acc = 0
    baseline_param={}
    es_list=list(np.arange(100,400,5))
    for es in es_list:
        tmp_acc = train_baseline(es)
        if np.mean(tmp_acc)>baseline_best_acc:
            baseline_best_acc = np.mean(tmp_acc)
            baseline_acc=tmp_acc
            baseline_param={'es':str(es)}
    print('best_acc=',np.mean(baseline_acc),'(var=',np.var(baseline_acc),')\n')

    print('\tFor Withpsd:')
    withpsd_acc =[]
    withpsd_best_acc = 0
    withpsd_param={}
    for es in es_list:
        tmp_acc = train_withpsd(es)
        if np.mean(tmp_acc)>withpsd_best_acc:
            withpsd_best_acc = np.mean(tmp_acc)
            withpsd_acc=tmp_acc
            withpsd_param={'es':str(es)}
    print('best_acc=',np.mean(withpsd_acc),'(var=',np.var(withpsd_acc),')\n')

    print(ttest_rel(baseline_acc,withpsd_acc))
    baseline={'acc':baseline_acc,'param':baseline_param}
    withpsd={'acc':withpsd_acc,'param':withpsd_param}
    return baseline,withpsd

def MLP():
    print('\nstart MLP ......')
    def train_baseline(hidden, activate, solve, l2):
        tmp_acc = []
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
                tmp_acc.append(acc)

        print('hidden='+str(hidden)+',activate='+activate+',solve'+solve+',l2=',l2,
                ':\tacc=',np.mean(tmp_acc),'(var=',np.var(tmp_acc),')')
        return tmp_acc
    
    def train_withpsd(hidden, activate, solve, l2):
        tmp_acc = []
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
                tmp_acc.append(acc)

        print('hidden='+str(hidden)+',activate='+activate+',solve'+solve+',l2=',l2,
                ':\tacc=',np.mean(tmp_acc),'(var=',np.var(tmp_acc),')')
        return tmp_acc

    print('\tFor Baseline:')
    baseline_acc =[]
    baseline_best_acc = 0
    baseline_param={}
    hidden_list=[(100,),(50,),(30,),(10,),(10,10),(30,10),(10,30),(30,30),(25,10,5)]
    activate_list = activate_list = ['identity','tanh', 'relu']
    solve_list = ['lbfgs', 'adam']
    l2_list=[1e-6,1e-4,1e-2,0]
    for hidden in hidden_list:
        for activate in activate_list:
            for solve in solve_list:
                for l2 in l2_list:
                    tmp_acc = train_baseline(hidden, activate, solve, l2)
                    if np.mean(tmp_acc)>baseline_best_acc:
                        baseline_best_acc = np.mean(tmp_acc)
                        baseline_acc=tmp_acc
                        baseline_param={'hidden':str(hidden),'activate':activate,'solve':solve,'l2':str(l2)}
    print('best_acc=',np.mean(baseline_acc),'(var=',np.var(baseline_acc),')')

    print('\tFor Withpsd:')
    withpsd_acc =[]
    withpsd_best_acc = 0
    withpsd_param={}
    for hidden in hidden_list:
        for activate in activate_list:
            for solve in solve_list:
                for l2 in l2_list:
                    tmp_acc = train_withpsd(hidden, activate, solve, l2)
                    if np.mean(tmp_acc)>withpsd_best_acc:
                        withpsd_best_acc = np.mean(tmp_acc)
                        withpsd_acc=tmp_acc
                        withpsd_param={'hidden':str(hidden),'activate':activate,'solve':solve,'l2':str(l2)}
    print('best_acc=',np.mean(withpsd_acc),'(var=',np.var(withpsd_acc),')\n')

    print(ttest_rel(baseline_acc,withpsd_acc))
    baseline={'acc':baseline_acc,'param':baseline_param}
    withpsd={'acc':withpsd_acc,'param':withpsd_param}
    return baseline,withpsd



SVM_baseline, SVM_withpsd = SVM()
GBDT_baseline, GBDT_withpsd = GBDT()
RF_baseline, RF_withpsd = RF()
MLP_baseline, MLP_withpsd = MLP()


results={'SVM_baseline':SVM_baseline, 'SVM_withpsd':SVM_withpsd,
        'GBDT_baseline':GBDT_baseline,'GBDT_withpsd':GBDT_withpsd,
        'RF_baseline':RF_baseline, 'RF_withpsd':RF_withpsd,
        'MLP_baseline':MLP_baseline, 'MLP_withpsd':MLP_withpsd}
import json
with open('SVM.json', 'w') as f:
  json.dump(results, f)