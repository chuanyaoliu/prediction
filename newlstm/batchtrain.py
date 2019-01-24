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
from encodermodel import AutoEncoder

batch_size = 32
sequence_len=30

autoencoder = torch.load('./model/encoder_epoch90.pkl')

maxNum,minNum = findMaxMin('CMAPSSData/train_FD001.txt')

data = readFile('CMAPSSData/train_FD001.txt')
data = selectFeature(data,maxNum,minNum)
dataX,dataY,dataAxis = makeUpSeq(data,sequence_len)
#dataX,dataY,dataAxis = makeUpEncoder(data,sequence_len)


testdata = readFile('CMAPSSData/test_FD001.txt')
testdata = selectFeature(testdata,maxNum,minNum)
testrul = readrul('CMAPSSData/RUL_FD001.txt')
testdataX,testdataY,testdataAxis = makeUpSeq(testdata,sequence_len,testrul)
#testdataX,testdataY,testdataAxis = makeUpSeq2(autoencoder,testdata,sequence_len,testrul)

dataX = torch.FloatTensor(dataX[:len(dataX)-len(dataX)%batch_size])
dataY = torch.FloatTensor(dataY[:len(dataX)-len(dataX)%batch_size])
dataAxis = torch.FloatTensor(dataAxis[:len(dataX)-len(dataX)%batch_size])
torch_dataset = Data.TensorDataset(dataX, dataY,dataAxis)

# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)

testdataX = torch.FloatTensor(testdataX[:len(testdataX)-len(testdataX)%batch_size])
testdataY = torch.FloatTensor(testdataY[:len(testdataX)-len(testdataX)%batch_size])
testdataAxis = torch.FloatTensor(testdataAxis[:len(testdataX)-len(testdataX)%batch_size])
torch_testdataset = Data.TensorDataset(testdataX, testdataY,testdataAxis)

# 把 dataset 放入 DataLoader
testloader = Data.DataLoader(
    dataset=torch_testdataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)

net = Net(16, 200, sequence_len,batch_size)
#net = torch.load('./model130/model_epoch00012.pkl')
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  
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
for ep in range(100):
    print ep
    X=[]
    Y=[]
    Z=[]
    lossAll = []
    for step,(x,y,axis) in enumerate(loader):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        
        prediction = net(Variable(x))   
        loss = loss_func(prediction[:,-1],y[:,-1])
        optimizer.zero_grad()  
        loss.backward()        
        optimizer.step()
        if step%100==0:
            print step,loss
        #if ep==epoch-1:
        X.append(-axis[:,-1])
        Y.append(y.detach().numpy()[:,-1])
        Z.append(prediction.detach().numpy()[:,-1])
        lossAll.append(y.detach().numpy()[:,-1]-prediction.detach().numpy()[:,-1])
    
    
    print len(lossAll),calMSE(lossAll)
    
    if ep%1==0:
        model_name = "./model130/model_epoch00000"+str(ep)+".pkl"
        #torch.save(net, model_name)
        print model_name,"has been saved"
    '''
    plt.scatter(X, Y,s=10)
    plt.scatter(X, Z,s=10)
    plt.show()
    '''
    testMSE(testdataX,testdataY,testdataAxis)
