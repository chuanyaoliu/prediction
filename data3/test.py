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

maxNum,minNum = findMaxMin('CMAPSSData/train_FD003.txt')


testdata = readFile('CMAPSSData/test_FD003.txt')
testdata = selectFeature(testdata,maxNum,minNum)
testrul = readrul('CMAPSSData/RUL_FD003.txt')
testdataX,testdataY,testdataAxis = makeUpSeqTest(testdata,sequence_len,testrul)

#net = Net(1, 300, sequence_len,batch_size)
net = torch.load('./model/best_17.5.pkl')
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
        encoded,decoded,prediction = net(Variable(x))
        loss = loss_func(prediction,torch.FloatTensor(y))
        i+=1
        res+= loss.detach().numpy() 
        if i%100==0:
            print i,prediction,y
            print 'test',pow(res/i,0.5)
    print 'test',pow(res/i,0.5)      
          

    
testMSE(testdataX,testdataY,testdataAxis)   
'''
for ep in range(40):
    print ep
    i = 0
    lossAll = []
    for (x,y,axis) in zip(dataX,dataY,dataAxis):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        
        encoded,decoded,prediction = net(Variable(x)) 
        loss1 = loss_func(prediction[-1],y[-1])
        loss2 = loss_func(decoded,x)
        loss = loss1+loss2
        optimizer.zero_grad()  
        loss1.backward()        
        optimizer.step()
        lossAll.append(loss1.detach().numpy())
        i=i+1 
        if i%1000==0:
            print i,loss1,loss2,prediction,y
            res = 0
            for x in lossAll:
                res = res+x
            print 'train',len(lossAll),pow(res/len(lossAll),0.5)
            
        
    res = 0
    for x in lossAll:
        res = res+x
    print 'train',len(lossAll),pow(res/len(lossAll),0.5)
    model_name = "./model/model_epoch"+str(ep)+".pkl"
    torch.save(net, model_name)
    print model_name,"has been saved"
    testMSE(testdataX,testdataY,testdataAxis)

        
    #plt.scatter(X, Y,s=10)
    #plt.scatter(X, Z,s=10)
    #plt.plot(X,E)
    #plt.show()
'''   
    
