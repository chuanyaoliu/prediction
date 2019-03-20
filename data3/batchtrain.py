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
import torch.nn.functional as F    
from torch.autograd import Variable
import torch.utils.data as Data
from util import * 
from model import Net 

batch_size = 32
sequence_len=30

maxNum,minNum = findMaxMin('CMAPSSData/train_FD001.txt')

data = readFile('CMAPSSData/train_FD001.txt')
data = selectFeature(data,maxNum,minNum)
dataX,dataY,dataAxis = makeUpSeq(data,sequence_len)

testdata = readFile('CMAPSSData/test_FD001.txt')
testdata = selectFeature(testdata,maxNum,minNum)
testrul = readrul('CMAPSSData/RUL_FD001.txt')
testdataX,testdataY,testdataAxis = makeUpSeq(testdata,sequence_len,testrul)


dataX.extend(dataX[:(batch_size-len(dataX)%batch_size)])
dataY.extend(dataY[:(batch_size-len(dataX)%batch_size)])
dataAxis.extend(dataAxis[:(batch_size-len(dataX)%batch_size)])

dataX = torch.FloatTensor(dataX[:(len(dataX)-len(dataX)%batch_size)])
dataY = torch.FloatTensor(dataY[:(len(dataX)-len(dataX)%batch_size)])
dataAxis = torch.FloatTensor(dataAxis[:(len(dataX)-len(dataX)%batch_size)])
torch_dataset = Data.TensorDataset(dataX,dataY,dataAxis)

loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=False,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)

testdataX = torch.FloatTensor(testdataX[:len(testdataX)-len(testdataX)%batch_size])
testdataY = torch.FloatTensor(testdataY[:len(testdataX)-len(testdataX)%batch_size])
testdataAxis = torch.FloatTensor(testdataAxis[:len(testdataX)-len(testdataX)%batch_size])
torch_testdataset = Data.TensorDataset(testdataX, testdataY,testdataAxis)

testloader = Data.DataLoader(
    dataset=torch_testdataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)

#net = Net(1, 300, sequence_len,batch_size)
net = torch.load('./model130/model_batch15.pkl')
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)  
loss_func = torch.nn.MSELoss()   

def testMSE(dataX,dataY,dataAxis):
    X=[]
    Y=[]
    Z=[]
    lossAll=[]
    a=0
    b=0
    c=0
    d=0
    for step,(x,y,axis) in enumerate(testloader):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        prediction = net(Variable(x))
        X.append(-axis[-1])
        Y.append(y[-1])
        Z.append(prediction.detach().numpy()[-1])
        lossAll.append(y.detach().numpy()[:,-1]-prediction.detach().numpy()[:,-1])
        '''
        if -axis[-1]!=a+1:
            X.append(a)
            Y.append(b)
            Z.append(c)
            lossAll.append(d)
        a=-axis[-1]
        b=y[-1]
        c=prediction.detach().numpy()[-1]
        d=y[-1]-prediction.detach().numpy()[-1]
        '''
    
    print len(lossAll),calMSE(lossAll)
    '''
    plt.scatter(X, Y,s=10)
    plt.scatter(X, Z,s=10)
    plt.show()
    '''
#testMSE(testdataX,testdataY,testdataAxis)   
for ep in range(50):
    print ep
    X=[]
    Y=[]
    Z=[]
    E=[]
    lossAll = []
    for step,(x,y,axis) in enumerate(loader):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        encoded,decoded,prediction = net(Variable(x))   
        loss1 = loss_func(prediction[:,-1],y[:,-1])
        loss2 = loss_func(decoded[:,-1],x[:,-1])*100
        loss = loss1+loss2
        
        optimizer.zero_grad()  
        loss.backward()        
        optimizer.step()
        if step%100==0:
            print step,loss1,loss2
        #if ep==epoch-1:
        '''
        X.append(-axis[:,-1])
        Y.append(y.detach().numpy()[:,-1])
        Z.append(prediction.detach().numpy()[:,-1])
        E.append(encoded.detach().numpy()[:,-1])
        '''
        lossAll.append(y.detach().numpy()[:,-1]-prediction.detach().numpy()[:,-1])
    print y.detach().numpy()[:,-1],prediction.detach().numpy()[:,-1]
    
    
    print len(lossAll),calMSE(lossAll)
    
    if ep%1==0:
        model_name = "./model130/model_batch"+str(ep)+".pkl"
        torch.save(net, model_name)
        print model_name,"has been saved"
    '''
    plt.scatter(X, Y,s=10)
    plt.scatter(X, Z,s=10)
    plt.show()
    '''
    #testMSE(testdataX,testdataY,testdataAxis)
    
