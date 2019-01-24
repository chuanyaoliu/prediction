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
from util import * 
from model import Net 
from encodermodel import AutoEncoder
sequence_len=30

#autoencoder = torch.load('./model/encoder_epoch0.pkl')

maxNum,minNum = findMaxMin('CMAPSSData/train_FD001.txt')

data = readFile('CMAPSSData/train_FD001.txt')
data = selectFeature(data,maxNum,minNum)
dataX,dataY,dataAxis = makeUpSeq(data,sequence_len)
#dataX,dataY,dataAxis = makeUpSeq2(autoencoder,data,sequence_len)

testdata = readFile('CMAPSSData/test_FD001.txt')
testdata = selectFeature(testdata,maxNum,minNum)
testrul = readrul('CMAPSSData/RUL_FD001.txt')
testdataX,testdataY,testdataAxis = makeUpSeq(testdata,sequence_len,testrul)
#testdataX,testdataY,testdataAxis = makeUpSeq2(autoencoder,testdata,sequence_len,testrul)

#net = Net(16, 300, sequence_len)
net = torch.load('./model130/model_epoch00012.pkl')
optimizer = torch.optim.SGD(net.parameters(), lr=0.00001)  
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
    for (x,y,axis) in zip(dataX,dataY,dataAxis):
        x = torch.FloatTensor(x)
        prediction = net(Variable(x))
        
        X.append(-axis[-1])
        Y.append(y[-1])
        Z.append(prediction.detach().numpy()[-1])
        lossAll.append(y[-1]-prediction.detach().numpy()[-1])
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
    res = 0
    for x in lossAll:
        res = res+pow(x,2)
    print len(lossAll),pow(res/len(lossAll),0.5)
    '''
    plt.scatter(X, Y,s=10)
    plt.scatter(X, Z,s=10)
    plt.show()
    '''
#testMSE(testdataX,testdataY,testdataAxis)   
for ep in range(10):
    print ep
    i = 0
    X=[]
    Y=[]
    Z=[]
    lossAll = []
    for (x,y,axis) in zip(dataX,dataY,dataAxis):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        
        prediction = net(Variable(x))   
        loss = loss_func(prediction[-1],y[-1])
        optimizer.zero_grad()  
        loss.backward()        
        optimizer.step()
        i=i+1 
        if i%1000==0:
            print i,loss
        #if ep==epoch-1:
        X.append(-axis[-1])
        Y.append(y.detach().numpy()[-1])
        Z.append(prediction.detach().numpy()[-1])
        lossAll.append(y.detach().numpy()[-1]-prediction.detach().numpy()[-1])
    res = 0
    for x in lossAll:
        res = res+pow(x,2)
    print len(lossAll),pow(res/len(lossAll),0.5)
    
    if ep%1==0:
        model_name = "./model130/model_epoch0000"+str(ep+12)+".pkl"
        torch.save(net, model_name)
        print model_name,"has been saved"
    '''
    plt.scatter(X, Y,s=10)
    plt.scatter(X, Z,s=10)
    plt.show()
    '''
    testMSE(testdataX,testdataY,testdataAxis)
