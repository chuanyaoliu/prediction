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

#net = Net(3, 300, sequence_len,batch_size)
#net = torch.load('./model130/model_epoch004.pkl')
net = torch.load('./modelNew/model_epoch29.pkl')
#net = torch.load('./modelNew/model3_epoch18.pkl')
#optimizer = torch.optim.SGD(net.parameters(), lr=0.001)  

conv5_params = list(map(id, net.encoder.parameters()))
base_params = filter(lambda p: id(p) not in conv5_params,
                     net.parameters())
conv5_params = list(map(id, net.decoder.parameters()))
base_params = filter(lambda p: id(p) not in conv5_params,
                     base_params)
optimizer = torch.optim.SGD([
            {'params': base_params},
            {'params': net.encoder.parameters(), 'lr': 0.000001},
            {'params': net.decoder.parameters(), 'lr': 0.000001}],
            lr=0.000001,momentum=0.9)  
           
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
for ep in range(40):
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
        loss1 = loss_func(prediction,y)/10
        loss2 = loss_func(decoded,x)*1000
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
        model_name = "./modelNew/model3_epoch"+str(ep+19)+".pkl"
        torch.save(net, model_name)
        print model_name,"has been saved"
        #plt.scatter(X, Y,s=10)
        #plt.scatter(X, Z,s=10)
        #plt.plot(X,E)
        #plt.show()
    
    

