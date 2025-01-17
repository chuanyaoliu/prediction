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
a = [0,1,0,0,0,0,0.3,0.2,0.5,0,0,0.6,0.1,0.1,0,0.4,0.5,0.8,0.2,0.3,0,0.4,0,0,0.9,0.7]
a = np.asarray(a)
print [a[i] for i in[6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,24,25]]
maxNum,minNum = findMaxMin2('CMAPSSData/train_FD002.txt')

data = readFile('CMAPSSData/train_FD002.txt')
data = selectFeature2(data,maxNum,minNum)
'''
for jj in range(26):
    X=[]
    Y=[]
    for x in range(len(data)):
        meanNum = np.mean(data[x],axis=0)
        stdNum = np.std(data[x],axis=0)
        for i in range(len(data[x])):
            X.append(data[x][i][1])
            Y.append(data[x][i][jj])
        
    plt.scatter(X, Y, s=15)
    #plt.scatter(X, Z, s=15)
    #plt.xlim(-10, 300)
    #plt.ylim(-10, 300)
    plt.show()
'''

dataX,dataY,dataAxis = makeUpSeq(data,sequence_len)

testdata = readFile('CMAPSSData/test_FD002.txt')
testdata = selectFeature2(testdata,maxNum,minNum)
testrul = readrul('CMAPSSData/RUL_FD002.txt')
testdataX,testdataY,testdataAxis = makeUpSeqTest3(testdata,sequence_len,testrul)

net = Net(3, 300, sequence_len,batch_size)
#net = torch.load('./model/model_elstm27.pkl')
#net = torch.load('./modelZ/model_epoch0.pkl')
#net = torch.load('./modelNew/model3_epoch18.pkl')

optimizer = torch.optim.SGD(net.parameters(), lr=0.00001)  
'''
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
            lr=0.00001)  
'''           
loss_func = torch.nn.MSELoss()   

def testMSE(dataX,dataY,dataAxis):
    i=0
    res=0
    for (x,y,axis) in zip(dataX,dataY,dataAxis):
        x = torch.FloatTensor(x)
        encoded,decoded,prediction = net(Variable(x),False)
        loss = loss_func(prediction[-1],torch.FloatTensor(y)[-1])
        i+=1
        res+= loss.detach().numpy() 
        '''
        if i%2000==0:
            print i,prediction,y
            print 'test',pow(res/i,0.5)
        '''
    print 'test',pow(res/i,0.5)
    
#testMSE(testdataX,testdataY,testdataAxis)   
for ep in range(40):
    print ep
    i = 0
    res = 0
    for (x,y,axis) in zip(dataX,dataY,dataAxis):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        
        encoded,decoded,prediction = net(Variable(x),True) 
        loss1 = loss_func(prediction,y)
        loss2 = loss_func(decoded,x)
        loss = loss1+loss2
        optimizer.zero_grad()  
        loss.backward()        
        optimizer.step()
        i=i+1 
        res+=loss1.detach().numpy() 
        if i%4000==0:
            print i,loss1,loss2,prediction,y
            print 'train',pow(res/i,0.5)
            
    print 'train',pow(res/i,0.5)
    model_name = "./model/model_epoch"+str(ep)+".pkl"
    torch.save(net, model_name)
    print model_name,"has been saved"
    testMSE(testdataX,testdataY,testdataAxis)

        
    #plt.scatter(X, Y,s=10)
    #plt.scatter(X, Z,s=10)
    #plt.plot(X,E)
    #plt.show()
    
    

