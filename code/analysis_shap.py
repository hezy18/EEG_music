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

import shap
import matplotlib.pyplot as plt

seeds = [1,2,3,4,5]
model_random_state=99

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
    print('start MLP ......')
    def train_baseline(hidden, activate, solve, l2):
        mse_list=[]
        for time in range(N_times):
            for fold in  range(N_split):
                train_index = train_all_index[time][fold]
                test_index = test_all_index[time][fold]
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = rating[train_index], rating[test_index]
                
                clf = MLPRegressor(hidden_layer_sizes = hidden, activation=activate, solver = solve, alpha=l2, max_iter = 1000,random_state=model_random_state)
            
                clf.fit(X_train, y_train)
                
                if time==0 and fold==0:
                    explainer = shap.KernelExplainer(model=clf.predict, data=X_train, link="identity")
                    shap_values = explainer.shap_values(X_train,nsamples=50)

                    #shap.initjs()
                    #shap.force_plot(explainer.expected_value, shap_values[0], X_test[:1],show=False)
                    
                    #plt.savefig('shap1.png')
                    shap.summary_plot(shap_values, X_train,max_display=50,show=False)
                    plt.savefig('fac_importance.pdf')
                y_pred = clf.predict(X_test)
            
                mse = mean_squared_error(y_test, y_pred)           
                mse_list.append(mse)

                if time==0 and fold==0:
                    print(mse)
                
                
                    
        print('mse=',np.mean(mse_list))
        return mse_list
    
    def train_withpsd(hidden, activate, solve, l2):
        mse_list=[]
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
                mse_list.append(mse)
                
                if time==0 and fold==0:
                    print(mse)
                    explainer = shap.KernelExplainer(model=clf.predict, data=X_train, link="identity")
                    shap_values = explainer.shap_values(X_train,nsamples=50)

                    #shap.force_plot(explainer.expected_value, shap_values[0], X_test[:1],show=False)
                    shap.summary_plot(shap_values, X_train,max_display=50,show=False)
                    plt.savefig('fac_importance2.pdf')


        print('mse=',np.mean(mse_list))
        return mse_list

    print('\tFor Baseline:')
    #train_baseline((30,10), 'tanh', 'sgd', 0.01)

    print('\tFor Withpsd:')
    train_withpsd((30, 30), 'identity', 'sgd', 0.01)

#MLP_coef()

def Clf_MLP_coef():
    print('\nstart MLP ......')
    def train_baseline(hidden, activate, solve, l2):
        acc_list=[]
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
                acc_list.append(acc)
                
                if time==0 and fold==0:
                    print(acc)
                    explainer = shap.KernelExplainer(model=clf.predict, data=X_train, link="identity")
                    shap_values = explainer.shap_values(X_train,nsamples=50)

                    #shap.force_plot(explainer.expected_value, shap_values[0], X_test[:1],show=False)
                    shap.summary_plot(shap_values, X_train,max_display=50,show=False)
                    plt.savefig('clf_importance.pdf')
        return acc_list
    
    def train_withpsd(hidden, activate, solve, l2):
        acc_list=[]
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
                acc_list.append(acc)            
                if time==0 and fold==0:
                    print(acc)
                    explainer = shap.KernelExplainer(model=clf.predict, data=X_train, link="identity")
                    shap_values = explainer.shap_values(X_train,nsamples=50)

                    #shap.force_plot(explainer.expected_value, shap_values[0], X_test[:1],show=False)
                    shap.summary_plot(shap_values, X_train,max_display=50,show=False)
                    plt.savefig('clf_importance2.pdf')
        print('acc=',np.mean(acc_list))
        return acc_list

    print('\tFor Baseline:')
    #hidden_list=[(100,),(50,),(30,),(10,),(10,10),(30,10),(10,30),(30,30),(25,10,5)]
    #activate_list = activate_list = ['identity','logistic','tanh', 'relu']
    #solve_list = ['lbfgs', 'adam','sgd']
    #l2_list=[1e-6,1e-4,1e-2,0]
    #train_baseline((100,), 'tanh', 'adam', 1e-06)
   


    print('\tFor Withpsd:')
    train_withpsd((10, 30), 'tanh', 'adam', 0.01)
   
                
Clf_MLP_coef()