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

sequence_len=30
batch_size = 1

maxNum,minNum = findMaxMin('CMAPSSData/train_FD001.txt')

data = readFile('CMAPSSData/train_FD001.txt')
data = selectFeature(data,maxNum,minNum)
dataX,dataY,dataAxis = makeUpSeq(data,sequence_len)

testdata = readFile('CMAPSSData/test_FD001.txt')
testdata = selectFeature(testdata,maxNum,minNum)
testrul = readrul('CMAPSSData/RUL_FD001.txt')
testdataX,testdataY,testdataAxis = makeUpSeq(testdata,sequence_len,testrul)

#net = Net(1, 300, sequence_len,batch_size)
net = torch.load('./model130/model_epoch004.pkl')
optimizer = torch.optim.SGD(net.parameters(), lr=0.00001)  
loss_func = torch.nn.MSELoss()   

def testMSE(dataX,dataY,dataAxis):
    X=[]
    Y=[]
    Z=[]
    lossAll=[]
    for (x,y,axis) in zip(dataX,dataY,dataAxis):
        x = torch.FloatTensor(x)
        encoded,decoded,prediction = net(Variable(x))
        '''
        X.append(-axis[-1])
        Y.append(y[-1])
        Z.append(prediction.detach().numpy()[-1])
        E.append(encoded.detach().numpy()[-1])
        '''
        lossAll.append(y[-1]-prediction.detach().numpy()[-1])
        
    res = 0
    for x in lossAll:
        res = res+pow(x,2)
    print 'test',len(lossAll),pow(res/len(lossAll),0.5)
    '''
    plt.scatter(X, Y,s=10)
    plt.scatter(X, Z,s=10)
    plt.plot(X,E)
    plt.show()
    '''
#testMSE(testdataX,testdataY,testdataAxis)   
for ep in range(100):
    print ep
    i = 0
    X=[]
    Y=[]
    Z=[]
    E=[]
    lossAll = []
    for (x,y,axis) in zip(dataX,dataY,dataAxis):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        
        encoded,decoded,prediction = net(Variable(x))   
        loss1 = loss_func(prediction,y)
        loss2 = loss_func(decoded,x)*100
        loss = loss1+loss2
        optimizer.zero_grad()  
        loss.backward()        
        optimizer.step()
        i=i+1 
        if i%1000==0:
            print i,loss1,loss2
            
        X.append(-axis[-1])
        Y.append(y.detach().numpy()[-1])
        Z.append(prediction.detach().numpy()[-1])
        E.append(encoded.detach().numpy()[-1])
        lossAll.append(y.detach().numpy()[-1]-prediction.detach().numpy()[-1])
    res = 0
    for x in lossAll:
        res = res+pow(x,2)
    print 'train',len(lossAll),pow(res/len(lossAll),0.5)
    testMSE(testdataX,testdataY,testdataAxis)
    if ep%1==0:
        model_name = "./model130/model_epoch00"+str(ep+5)+".pkl"
        torch.save(net, model_name)
        print model_name,"has been saved"
        #plt.scatter(X, Y,s=10)
        #plt.scatter(X, Z,s=10)
        #plt.plot(X,E)
        #plt.show()
    
    

