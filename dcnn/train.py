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
from cnn import Net 

filename = 'FD002'
sequence_len=30
batch_size = 1
'''
a = [0,1,0,0,0,0,0.3,0.2,0.5,0,0,0.6,0.1,0.1,0,0.4,0.5,0.8,0.2,0.3,0,0.4,0,0,0.9,0.7]
a = np.asarray(a)
print [a[i] for i in[6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,24,25]]
'''
maxNum,minNum = findMaxMin2('CMAPSSData/train_%s.txt' % filename)
#print maxNum,minNum
data = readFile('CMAPSSData/train_%s.txt'% filename)
data,condition = selectFeature2(data,maxNum,minNum)
print len(data),len(data[0]),len(condition),len(condition[0])
'''
for jj in range(26):
    X=[]
    Y=[]
    for x in range(len(data)):
        meanNum = np.mean(data[x],axis=0)
        stdNum = np.std(data[x],axis=0)
        for i in range(len(data[x])):
            X.append(data[x][i][0])
            Y.append(condition[x][i][jj])
        
    plt.scatter(X, Y, s=15)
    #plt.scatter(X, Z, s=15)
    #plt.xlim(-10, 300)
    #plt.ylim(-10, 300)
    plt.show()
'''

dataX,dataY,dataAxis = makeUpSeq2(data,sequence_len)
data = None

testdata = readFile('CMAPSSData/test_%s.txt'% filename)
testdata,conditiontest = selectFeature2(testdata,maxNum,minNum)
testrul = readrul('CMAPSSData/RUL_%s.txt'% filename)
testdataX,testdataY,testdataAxis = makeUpSeq2(testdata,sequence_len,testrul)
testdata = None
net = Net(17,4, 300, sequence_len,batch_size)
#net = torch.load('./model/model_elstm27.pkl')
#net = torch.load('./model/data4_epoch301.pkl')
#net = torch.load('./modelNew/model3_epoch18.pkl')
learning_rate = 0.0001
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)  
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

def testMSE(dataX,dataY,dataAxis,condition):
    i=0
    res=0
    score=0
    for (x,y,axis,c) in zip(dataX,dataY,dataAxis,condition):
        x = torch.FloatTensor(x)
        c = torch.LongTensor(c)
        prediction = net(Variable(x),Variable(c),False)
        loss = loss_func(prediction[-1],torch.FloatTensor(y)[-1])
        i+=1
        res+= loss.detach().numpy() 
        score+= calScore(prediction[-1],y[-1])
        '''
        if i%2000==0:
            print i,prediction,y
            print 'test',pow(res/i,0.5)
        '''
    print 'test',pow(res/i,0.5),score
    
#testMSE(testdataX,testdataY,testdataAxis)   
for ep in range(40000):
    i = 0
    res = 0
    for (x,y,axis,c) in zip(dataX,dataY,dataAxis,condition):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        c = torch.LongTensor(c)
        prediction = net(Variable(x),Variable(c),True) 

        loss1 = loss_func(prediction,y[18:])
        #print loss1,loss2
        optimizer.zero_grad()  
        net.zero_grad()
        loss1.backward()        
        optimizer.step()
        i=i+1 
        res+=loss1.detach().numpy() 
        if i%5000==0:
            print i,loss1,prediction,y
            print 'train',pow(res/i,0.5)
    
    
    if ep!=0 and ep%50==0:
        print ep,loss1
        #if ep%1000==0:
        #    learning_rate /= 10        
        print 'train',pow(res/i,0.5)
        model_name = "./model/data4_epoch"+str(ep+1)+".pkl"
        torch.save(net, model_name)
        print model_name,"has been saved"
        testMSE(testdataX,testdataY,testdataAxis,conditiontest)

        
    #plt.scatter(X, Y,s=10)
    #plt.scatter(X, Z,s=10)
    #plt.plot(X,E)
    #plt.show()
    
    

