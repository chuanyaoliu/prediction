#coding:utf8
import codecs
import pdb
import sys
import copy
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F     # 激励函数都在这
from torch.autograd import Variable
import torch.utils.data as Data

'''
b = np.array([[[2,3,5],
             [2,3,5],
             [2,3,5]],
             [[2,3,5],
             [2,3,5],
             [2,3,5]]])
a = np.array([1,2,3])
print a[[0,2]]
'''
sequence_len = 30
BATCH_SIZE =1

data = []
oneData=[]
run = 1
with open('CMAPSSData/train_FD001.txt') as inp:
    for line in inp.readlines():
        line = line.split()
        if int(line[0])==run:
            oneData.append(map(eval,line))
        else:
            data.append(oneData)
            oneData=[]
            oneData.append(map(eval,line))
            run=int(line[0])
            if run==13:
                break
    data.append(oneData)

print 'train',len(data),len(data[0])

data = np.asarray(data)

for x in range(len(data)):
    meanNum = np.mean(data[x],axis=0)
    stdNum = np.std(data[x],axis=0)
    for i in range(len(stdNum)):
        if stdNum[i]==0:
            stdNum[i]=1
    meanNum[1]=0
    stdNum[1]=1
    data[x]= (data[x]-meanNum)/stdNum
    data[x] = data[x][:,[1,2,3,6,7,8,11,12,13,15,16,17,18,19,21,24,25]]




dataX = []
dataY = []
dataAxis = []
seqAxis = []
sequenceX = []
sequenceY = []
for i in range(len(data)):
    sequenceX = []
    sequenceY = []
    seqAxis = []
    for j in range(len(data[i])):
        x = data[i][j][1:]
        y = len(data[i])-data[i][j][0]
        seqAxis.append(y)          
        y = y if y<130 else 130
        sequenceX.append(x)
        sequenceY.append(y)
        if j>sequence_len-2:
            dataX.append(copy.deepcopy(sequenceX))
            dataY.append(copy.deepcopy(sequenceY))
            dataAxis.append(copy.deepcopy(seqAxis))
            sequenceX.pop(0)
            sequenceY.pop(0)
            seqAxis.pop(0)
                 

print len(dataX),len(dataY),len(dataAxis),len(dataX[0]),len(dataX[:len(dataX)-len(dataX)%BATCH_SIZE])




class Net(torch.nn.Module):  
    def __init__(self, n_feature, n_hidden,n_hidden2, n_output):
        super(Net, self).__init__()   
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)   
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)  
        self.hidden3 = torch.nn.Linear(n_hidden2,50)  
        self.hidden4 = torch.nn.Linear(50,12)  
        self.predict = torch.nn.Linear(300, n_output)   
        self.lstm = torch.nn.LSTM(input_size=n_feature,hidden_size=n_hidden/2,num_layers=1, bidirectional=True)
        self.rnn = torch.nn.RNN(n_hidden2,n_hidden2,1,True)
        self.batch=BATCH_SIZE
        self.dropout=torch.nn.Dropout(p=0.5)        
        self.n_hidden=n_hidden
        
    def init_hidden_lstm(self):
        return (torch.randn(2, self.batch, self.n_hidden/2),
                torch.randn(2, self.batch, self.n_hidden/2))
    def forward(self, x):   
        '''
        x = F.relu(self.hidden1(x))      
        x = F.relu(self.hidden2(x)) 
        x = F.relu(self.hidden3(x)) 
        x = F.relu(self.hidden4(x)) 
        '''
        self.hidden = self.init_hidden_lstm()
        x = x.view(sequence_len,self.batch,-1)
        x, self.hidden = self.lstm(x, self.hidden)
        '''
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x)) 
        x = F.relu(self.hidden4(x))
        '''
        #x = self.dropout(x)
        x = self.predict(x)   
        
        return torch.transpose(x,0,1).view(self.batch,-1)

net = Net(n_feature=16, n_hidden=300,n_hidden2=150, n_output=1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001) 
loss_func = torch.nn.MSELoss()     

# 先转换成 torch 能识别的 Dataset
dataX = torch.FloatTensor(dataX[:len(dataX)-len(dataX)%BATCH_SIZE])
dataY = torch.FloatTensor(dataY[:len(dataX)-len(dataX)%BATCH_SIZE])
dataAxis = torch.FloatTensor(dataAxis[:len(dataX)-len(dataX)%BATCH_SIZE])
torch_dataset = Data.TensorDataset(dataX, dataY,dataAxis)

# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)


X=[]
Y=[]
Z=[]
lossAll = []

epoch = 3
i = 0
for ep in range(epoch):
    print ep
    i = 0
    for step, (x,y,axis) in enumerate(loader):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        prediction = net(Variable(x))   
        loss = loss_func(prediction,y)/1
        optimizer.zero_grad()  
        loss.backward()        
        optimizer.step()
        i=i+1 
        if i%10==0:
            print i,loss
        if ep==epoch-1:
            X.append(axis.detach().numpy()[-1][-1])
            Y.append(y.detach().numpy()[-1][-1])
            Z.append(prediction.detach().numpy()[-1][-1])
        lossAll.append(loss)
    res=0
    for x in lossAll:
        res = res+x        
    print len(lossAll),res/len(lossAll)
    lossAll = []
''' 
#lossAll = np.asarray(lossAll)
#meanNum = np.mean(lossAll)
res = 0
for x in lossAll:
    res = res+pow(x,2)
print len(lossAll),res/len(lossAll)
lossAll = []
'''


def testMSE():
    testdata = []
    oneData=[]
    run = 1
    with open('CMAPSSData/test_FD001.txt') as inp:
        for line in inp.readlines():
            line = line.split()
            if int(line[0])==run:
                oneData.append(map(eval,line))
            else:
                testdata.append(oneData)
                oneData=[]
                oneData.append(map(eval,line))
                run=int(line[0])
                if run==13:
                    break
        testdata.append(oneData)

    print 'test',len(testdata),len(testdata[0])

    testdata = np.asarray(testdata)

    for x in range(len(testdata)):
        meanNum = np.mean(testdata[x],axis=0)
        stdNum = np.std(testdata[x],axis=0)
        for i in range(len(stdNum)):
            if stdNum[i]==0:
                stdNum[i]=1
        meanNum[1]=0
        stdNum[1]=1
        testdata[x]= (testdata[x]-meanNum)/stdNum
        testdata[x] = testdata[x][:,[1,2,3,6,7,8,11,12,13,15,16,17,18,19,21,24,25]]

    testrul=[]
    with open('CMAPSSData/RUL_FD001.txt') as inp:
        for line in inp.readlines():
            testrul.append(int(line))
    print 'testrul',len(testrul)
    
    
    dataX = []
    dataY = []
    dataAxis = []
    seqAxis = []
    sequenceX = []
    sequenceY = []
    for i in range(len(testdata)):
        sequenceX = []
        sequenceY = []
        seqAxis = []
        for j in range(len(testdata[i])):
            x = testdata[i][j][1:]
            y = testrul[i]+len(testdata[i])-testdata[i][j][0]
            seqAxis.append(y)          
            y = y if y<130 else 130
            sequenceX.append(x)
            sequenceY.append(y)
            if j>sequence_len-2:
                dataX.append(copy.deepcopy(sequenceX))
                dataY.append(copy.deepcopy(sequenceY))
                dataAxis.append(copy.deepcopy(seqAxis))
                sequenceX.pop(0)
                sequenceY.pop(0)
                seqAxis.pop(0)
    print len(dataX),len(dataY),len(dataAxis),len(dataX[0])   
    
    X=[]
    Y=[]
    Z=[]
    lossAll = []
    for (x,y,axis) in zip(dataX,dataY,dataAxis):
        x = torch.FloatTensor(x)
        prediction = net(Variable(x))

        X.append(axis[-1])
        Y.append(y[-1])
        Z.append(prediction.detach().numpy()[-1])
        lossAll.append(y[-1]-prediction.detach().numpy()[-1])

    lossAll = np.asarray(lossAll)
    res = 0
    for x in lossAll:
        res = res+pow(x,2)
    print len(lossAll),res/len(lossAll)
    plt.scatter(X, Y, s=15)
    plt.scatter(X, Z, s=15)
    plt.xlim(-10, 300)
    plt.ylim(-10, 300)
    plt.show()
    

#testMSE()

    

plt.scatter(X, Y, s=15)
plt.scatter(X, Z, s=15)
plt.xlim(-10, 300)
plt.ylim(-10, 300)
plt.show()

