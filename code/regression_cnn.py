import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math

import numpy as np
from sklearn import preprocessing
label = np.load('rating.npy')
data= np.load('data.npy')
data_withpsd = np.load('data_withpsd.npy')
print('n_cases=%d, n_features of baselin=%d, n_features of withpsd=%d'%(data.shape[0],data.shape[1],data_withpsd.shape[1]))

min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data)
min_max_scaler = preprocessing.MinMaxScaler()
data_withpsd = min_max_scaler.fit_transform(data_withpsd)

from torch.utils.data import Dataset
import torch

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

seeds = [1,2,3,4,5]
set_seed=2022

class_sample_counts = [35, 62, 112,77,33]
weights = 1./ torch.tensor(class_sample_counts, dtype=torch.float)

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

import random
torch.manual_seed(set_seed)
torch.cuda.manual_seed_all(set_seed)
np.random.seed(set_seed)
random.seed(set_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from scipy.stats import ttest_rel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, WeightedRandomSampler

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=5, stride =2)
        self.mp1 = nn.MaxPool1d(kernel_size=2)
        self.bn1 = nn.BatchNorm1d(6)
        self.conv2 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(16)
        self.mp2 = nn.MaxPool1d(kernel_size=2)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        #print(x.shape)
        in_size = x.size(0)  # one batch
        x = torch.unsqueeze(x, 1)
        # x: batchsize * 32 * 768
        # x: batchsize * 27
        #print(x.shape)
        x = F.relu(self.mp1(self.conv1(x)))
        #print(x.shape)
        x = F.relu(self.mp2(self.conv2(x)))
        # x: batchsize
        #print(x.shape)
        x = x.view(in_size, -1)
        # x: batchsize *
        #print(x.shape)
        #x = self.fc1(x)
        x = self.fc2(x)
        # # x: batchsize*2
        return x

class Netpsd(nn.Module):
    def __init__(self):
        super(Netpsd, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=5, stride =2)
        self.mp1 = nn.MaxPool1d(kernel_size=2)
        self.bn1 = nn.BatchNorm1d(3)
        self.conv2 = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(3)
        self.mp2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc1 = nn.Linear(132, 11)
        self.fc2 = nn.Linear(66, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        #print(x.shape)
        in_size = x.size(0)  # one batch
        x = torch.unsqueeze(x, 1)
        # x: batchsize * 32 * 768
        # x: batchsize * 27
        #print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        #x = F.relu(self.mp2(self.conv2(x)))
        #print(x.shape)
        x = x.view(in_size, -1)
        # x: batchsize *
        #print(x.shape)
        #x = self.fc1(x)
        x = self.fc2(x)
        # # x: batchsize*2
        return x

criterion = nn.MSELoss()

#criterion = nn.SmoothL1Loss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train_epoch(epoch, model, train_dataloader, optimizer):
    model.train()
    acc_meter, loss_meter, it_count = 0, 0, 0
    for inputs, target in train_dataloader:
        inputs = inputs.to(device)
        target = target.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(inputs)
        #print(output,target)
        loss = criterion(output, target)

        '''for name, weight in model.named_parameters():	
            if weight.requires_grad:
                print("weight.grad:", weight.grad.mean(), weight.grad.min(), weight.grad.max())
'''
        loss.backward()
        
        optimizer.step()
        
        loss_meter += loss.item() 
        it_count += 1
        acc = mean_squared_error(output.data.cpu(), target.cpu())
        acc_meter+=acc
        #print(epoch, loss, acc)
    return  loss_meter / it_count, acc_meter / it_count

def test_epoch(epoch, model, test_dataloader,optimizer):
    model.eval()
    acc_meter, loss_meter, it_count = 0, 0, 0
    with torch.no_grad():
        for inputs, target in test_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            
            loss = criterion(output, target)
            
            loss_meter += loss.item()
            it_count += 1

            acc = mean_squared_error(output.data.cpu(), target.cpu())
            acc_meter+=acc

    return  loss_meter / it_count, acc_meter / it_count


class CNNDataset(Dataset):
    def __init__(self, is_train, mode, idx):
        super(CNNDataset , self).__init__()
        self.is_train = is_train
        self.data = idx
        self.mode = mode
    
    def __getitem__(self,index):
        if self.mode == 1:
            x = data[self.data[index]]
        else:
            x = data_withpsd[self.data[index]]
        x = torch.tensor(x.copy(), dtype = torch.float)
        target = label[self.data[index]]
        '''y_true=torch.zeros(3)
        y_true[int(target)]=1
        # y_true= torch.eye(3)[target,:]'''
        y = torch.tensor(target, dtype=torch.float)
        return x,y
    
    def __len__(self):
        return len(self.data)

def baseline():
    acc_list=[] #25
    for time in range(N_times):
        for fold in  range(N_split):
            one_list=[] # 100
            train_index = train_all_index[time][fold]
            test_index = test_all_index[time][fold]
            train_targets = label[train_index]
            samples_weights = weights[train_targets-1]
            sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
            train_all_dataset=CNNDataset(is_train=True, mode = 2, idx = train_index)
            traindataset_loader = torch.utils.data.DataLoader(dataset=train_all_dataset, batch_size=32, shuffle=False,  sampler = sampler, num_workers=4)
            test_all_dataset=CNNDataset(is_train=False, mode = 1, idx = test_index)
            testdataset_loader = torch.utils.data.DataLoader(dataset=test_all_dataset, batch_size=16, shuffle=False, num_workers=4)

            model = Net().to(device)
            print('num_paramers',sum(p.numel() for p in model.parameters()))
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            for epoch in range(100):
                if epoch==50:
                    adjust_learning_rate(optimizer, 0.001)
                if epoch==80:
                    adjust_learning_rate(optimizer, 0.0001)
                train_loss, train_acc = train_epoch(epoch,model,traindataset_loader,optimizer)
                test_loss, test_acc = test_epoch(epoch,model,testdataset_loader,optimizer)
                print('epoch=',epoch,'train_loss=',train_loss,'train_mse=',train_acc,
                                    'test_loss=',test_loss,'train_mse=',test_acc)
                one_list.append(test_acc)
            acc_list.append(one_list)
    return acc_list
                
def withpsd():
    acc_list=[] #25
    for time in range(N_times):
        for fold in  range(N_split):
            one_list=[] # 100
            train_index = train_all_index[time][fold]
            test_index = test_all_index[time][fold]
            train_targets = label[train_index]
            samples_weights = weights[train_targets-1]
            sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
            train_all_dataset=CNNDataset(is_train=True, mode = 2, idx = train_index)
            traindataset_loader = torch.utils.data.DataLoader(dataset=train_all_dataset, batch_size=32, shuffle=False,  sampler = sampler, num_workers=4)
            test_all_dataset=CNNDataset(is_train=False, mode = 2, idx = test_index)
            testdataset_loader = torch.utils.data.DataLoader(dataset=test_all_dataset, batch_size=16, shuffle=False, num_workers=4)

            model = Netpsd().to(device)
            print('num_paramers',sum(p.numel() for p in model.parameters()))
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            for epoch in range(100):
                if epoch==50:
                    adjust_learning_rate(optimizer, 0.001)
                if epoch==80:
                    adjust_learning_rate(optimizer, 0.0001)
                train_loss, train_acc = train_epoch(epoch,model,traindataset_loader,optimizer)
                test_loss, test_acc = test_epoch(epoch,model,testdataset_loader,optimizer)
                print('epoch=',epoch,'train_loss=',train_loss,'train_mse=',train_acc,
                                    'test_loss=',test_loss,'train_mse=',test_acc)
                one_list.append(test_acc)
            acc_list.append(one_list)
    return acc_list
            

if __name__ == '__main__':
    
    print('for baseline')
    #baseline_acc = baseline()
        
    print('for withpsd')
    withpsd_acc = withpsd()


    #baseline_acc = np.array(baseline_acc)
    withpsd_acc = np.array(withpsd_acc)
    best_baseline = []
    best_bsl_acc=100
    best_withpsd = []
    best_psd_acc=100
    baseline_epoch=0
    for epoch in range(100):
        '''if np.mean(baseline_acc[:,epoch])<best_bsl_acc:
            best_bsl_acc=np.mean(baseline_acc[:,epoch])
            best_baseline = baseline_acc[:,epoch]
            baseline_epoch=epoch'''
        if np.mean(withpsd_acc[:,epoch])<best_psd_acc:
            best_psd_acc=np.mean(withpsd_acc[:,epoch])
            best_withpsd = withpsd_acc[:,epoch]
            withpsd_epoch=epoch
    print('baseline:best_acc=',np.mean(best_baseline),'(var=',np.var(best_baseline),')\n')
    print('withpsd:best_acc=',np.mean(best_withpsd),'(var=',np.var(best_withpsd),')\n')
    results={'cnn_withpsd':list(best_withpsd), 'cnn_baseline':list(best_baseline),'baselineepoch':str(baseline_epoch),'withpsdepoch':str(withpsd_epoch)}
    import json
    with open('cnn_bn.json', 'w') as f:
        json.dump(results, f)
    # print(ttest_rel(best_baseline,best_withpsd))
    