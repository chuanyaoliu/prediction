#coding:utf8
import codecs
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F     # 激励函数都在这
from torch.autograd import Variable

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
            #if run==3:
            #    break
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
            #if run==3:
            #    break
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
print 'testrul',len(testrul),testrul

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
        self.batch=1
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
        x = x.view(1,self.batch,-1)
        x, self.hidden = self.lstm(x, self.hidden)
        '''
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x)) 
        x = F.relu(self.hidden4(x))
        '''
        x = self.dropout(x)
        x = self.predict(x)        
        return x[0][0]

net = Net(n_feature=16, n_hidden=300,n_hidden2=150, n_output=1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)  
loss_func = torch.nn.MSELoss()     

X=[]
Y=[]
Z=[]
lossAll = []
minNum =100
epoch = 10
for ep in range(epoch):
    print ep
    for i in range(len(data)):
        for j in range(len(data[i])):
            x = torch.FloatTensor(data[i][j][1:])
            y = len(data[i])-data[i][j][0]            
            y = y if y<130 else 130
            y = torch.FloatTensor([y])
            prediction = net(Variable(x))    
            loss = loss_func(prediction,y)/100
            #loss = sqrt(loss)
            optimizer.zero_grad()  
            loss.backward()        
            optimizer.step()
            lossAll.append(y.item()-prediction.item())       
            #print loss
            
            #if ep==epoch-1:
                #X.append(len(data[i])-j)
                #Y.append(y.item())
                #Z.append(prediction.item())
            
    lossAll = np.asarray(lossAll)
    meanNum = np.mean(lossAll)
    res = 0
    for x in lossAll:
        res = res+pow(x-meanNum,2)
    print meanNum,len(lossAll),res/len(lossAll)
    lossAll = []
    
    for i in range(len(testdata)):
        for j in range(len(testdata[i])):
            x = torch.FloatTensor(testdata[i][j][1:])   
            prediction = net(Variable(x))            
            if j==len(testdata[i])-1:
                #print prediction,testrul[i]
                lossAll.append(testrul[i]-prediction.item())       

            if ep==epoch-1:
                X.append(len(testdata[i])-j)
                Y.append(testrul[i]-len(testdata[i])+j)
                Z.append(prediction.item())
    lossAll = np.asarray(lossAll)
    meanNum = np.mean(lossAll)
    res = 0
    for x in lossAll:
        res = res+pow(x-meanNum,2)
    print meanNum,len(lossAll),res/len(lossAll)
    lossAll = []
    
    
'''
plt.scatter(X, Y, s=15)
plt.scatter(X, Z, s=15)
plt.xlim(-10, 300)
plt.ylim(-10, 300)
plt.show()
'''
