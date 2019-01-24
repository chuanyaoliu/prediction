#coding:utf8
import codecs
import pdb
import sys
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F     # 激励函数都在这
from torch.autograd import Variable


'''
b = np.array([[[2,3,5],
             [6,4,7],
             [8,0,9]],
             [[2,3,5],
             [2,3,5],
             [2,3,5]]])
a = np.array([1,2,3])
print np.max(b[0],axis=0)
'''
sequence_len = 30


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
    maxNum = np.max(data[x],axis=0)
    minNum = np.min(data[x],axis=0)
    
    for i in range(len(maxNum)):
        if maxNum[i]==minNum[i]:
            maxNum[i]=maxNum[i]+1
    minNum[1]=0
    maxNum[1]=1
    data[x]= (data[x]-minNum)/(maxNum-minNum)
    data[x] = data[x][:,[1,2,3,6,7,8,11,12,13,15,16,17,18,19,21,24,25]]
#print data[0]

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(16, 200),
            nn.Tanh(),
            nn.Linear(200, 50),
            nn.Tanh(),
            nn.Linear(50, 1),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 200),
            nn.Tanh(),
            nn.Linear(200, 16),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

#autoencoder = AutoEncoder()
autoencoder = torch.load('./model/encoder2_epoch200.pkl')



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
        x = torch.FloatTensor(x)
        y = len(data[i])-data[i][j][0]
        encoded, decoded = autoencoder(Variable(x))
        seqAxis.append(y)          
        y = y if y<130 else 130
        sequenceX.append(encoded.item())
        sequenceY.append(y)
        if j>sequence_len-2:
            dataX.append(copy.deepcopy(sequenceX))
            dataY.append(copy.deepcopy(sequenceY))
            dataAxis.append(copy.deepcopy(seqAxis))
            sequenceX.pop(0)
            sequenceY.pop(0)
            seqAxis.pop(0)
                 

print len(dataX),len(dataY),len(dataAxis),len(dataX[0])




class Net(torch.nn.Module):  
    def __init__(self, n_feature, n_hidden,n_hidden2, n_output):
        super(Net, self).__init__()   
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden2)   
        self.hidden2 = torch.nn.Linear(n_hidden2, 50)  
        self.hidden3 = torch.nn.Linear(50,1)  
        self.predict = torch.nn.Linear(n_hidden, n_output)   
        self.lstm = torch.nn.LSTM(input_size=1,hidden_size=n_hidden/2,num_layers=1, bidirectional=True)
        self.rnn = torch.nn.RNN(n_hidden2,n_hidden2,1,True)
        self.batch=1
        self.dropout=torch.nn.Dropout(p=0.5)        
        self.n_hidden=n_hidden
        self.feature_3_embeds = torch.nn.Embedding(13,10)
        self.feature_21_embeds = torch.nn.Embedding(13,10)
    def init_hidden_lstm(self):
        return (torch.randn(2, self.batch, self.n_hidden/2),
                torch.randn(2, self.batch, self.n_hidden/2))
    def forward(self, x):   
        '''
        x = F.relu(self.hidden1(x))      
        x = F.relu(self.hidden2(x)) 
        x = F.relu(self.hidden3(x)) 
        '''
        self.hidden = self.init_hidden_lstm()
        
        #print x.size()
        x = x.view(sequence_len,self.batch,-1)
        x, self.hidden = self.lstm(x, self.hidden)
        '''
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x)) 
        x = F.relu(self.hidden4(x))
        '''
        #x = self.dropout(x)
        x = self.predict(x)   
        return x.view(-1)

net = Net(n_feature=1, n_hidden=200,n_hidden2=150, n_output=1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)  
loss_func = torch.nn.MSELoss()     









def testData():
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
        maxNum = np.max(testdata[x],axis=0)
        minNum = np.min(testdata[x],axis=0)
    
        for i in range(len(maxNum)):
            if maxNum[i]==minNum[i]:
                maxNum[i]=maxNum[i]+1
        minNum[1]=0
        maxNum[1]=1
        testdata[x]= (testdata[x]-minNum)/(maxNum-minNum)
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
            x = torch.FloatTensor(x)
        
            encoded, decoded = autoencoder(Variable(x))
            y = testrul[i]+len(testdata[i])-testdata[i][j][0]
            seqAxis.append(y)          
            y = y if y<130 else 130
            sequenceX.append(encoded.item())
            sequenceY.append(y)
            if j>sequence_len-2:
                dataX.append(copy.deepcopy(sequenceX))
                dataY.append(copy.deepcopy(sequenceY))
                dataAxis.append(copy.deepcopy(seqAxis))
                sequenceX.pop(0)
                sequenceY.pop(0)
                seqAxis.pop(0)
    print len(dataX),len(dataY),len(dataAxis),len(dataX[0])   
    return dataX,dataY,dataAxis
def testMSE(dataX,dataY,dataAxis):
    X=[]
    Y=[]
    Z=[]
    lossAll = []
    for (x,y,axis) in zip(dataX,dataY,dataAxis):
        x = np.asarray(x)
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
    print 'test',len(lossAll),res/len(lossAll)
    
    plt.scatter(X, Y, s=10)
    plt.scatter(X, Z, s=10)
    plt.show()
    

test1,test2,test3 = testData()





X=[]
Y=[]
Z=[]
lossAll = []

epoch = 30
i = 0
for ep in range(epoch):
    print ep
    i = 0
    X=[]
    Y=[]
    Z=[]
    lossAll = []
    for (x,y,axis) in zip(dataX,dataY,dataAxis):
        x = np.asarray(x)
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        prediction = net(Variable(x))   
        loss = loss_func(prediction[-1],y[-1])
        optimizer.zero_grad()  
        loss.backward()        
        optimizer.step()
        i=i+1 
        if i%100==0:
            print i,loss
        #if ep==epoch-1:
        X.append(axis[-1])
        Y.append(y.detach().numpy()[-1])
        Z.append(prediction.detach().numpy()[-1])
        lossAll.append(y.detach().numpy()[-1]-prediction.detach().numpy()[-1])
    res = 0
    for x in lossAll:
        res = res+pow(x,2)
    print len(lossAll),res/len(lossAll)
    model_name = "./model130/model_epoch"+str(ep)+".pkl"
   # torch.save(net, model_name)
    print model_name,"has been saved"
    if ep%5==0:
        testMSE(test1,test2,test3)
        plt.scatter(X, Y, s=10)
        plt.scatter(X, Z, s=10)
        plt.show()
    
     
lossAll = np.asarray(lossAll)
#meanNum = np.mean(lossAll)
res = 0
for x in lossAll:
    res = res+pow(x,2)
print len(lossAll),res/len(lossAll)
lossAll = []







    
'''
plt.scatter(X, Y, s=15)
plt.scatter(X, Z, s=15)
plt.xlim(-10, 300)
plt.ylim(-10, 300)
plt.show()
'''
