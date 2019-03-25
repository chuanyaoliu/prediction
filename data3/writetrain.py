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

maxNum,minNum = findMaxMin('CMAPSSData/train_FD002.txt')

#data = readFile('CMAPSSData/train_FD001.txt')
#data = selectFeature(data,maxNum,minNum)
#dataX,dataY,dataAxis = makeUpSeq(data,sequence_len)

testdata = readFile('CMAPSSData/test_FD002.txt')
testdata = selectFeature(testdata,maxNum,minNum)
testrul = readrul('CMAPSSData/RUL_FD002.txt')

testdataX,testdataY,testdataAxis = makeUpSeqTest3(testdata,sequence_len,testrul)

#net = Net(1, 300, sequence_len,batch_size)
net = torch.load('./model/data2.pkl')
#net = torch.load('./modelNew/model_epoch29.pkl')
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
            {'params': net.encoder.parameters(), 'lr': 0.00001},
            {'params': net.decoder.parameters(), 'lr': 0.00001}],
            lr=0.00001,momentum=0.9)  
           
loss_func = torch.nn.MSELoss()   


from kalman_filter import KalmanFilter
from numpy import matrix, diag, random


def testMSE(dataX,dataY,dataAxis):
    dt = 0.05
    # Standard deviation of random accelerations
    sigma_a = 0.2
    # Standard deviation of observations
    sigma_z = 0.2
    # State vector: [[Position], [velocity]]
    X = matrix([[0.0], [0.0]])
    # Initial state covariance
    P = diag((0.0, 0.0))
    # Acceleration model
    G = matrix([[(dt ** 2) / 2], [dt]])
    # State transition model
    #F = matrix([[1, dt], [0, 1]])
    F = matrix([[1, 1], [0, 1]])
    # Observation vector
    Z = matrix([[0.0], [0.0]])
    # Observation model
    H = matrix([[1, 0], [0, 0]])
    # Observation covariance
    R = matrix([[sigma_z ** 2, 0], [0, 1]])
    # Process noise covariance matrix
    Q = G * (G.T) * sigma_a ** 2


    # Initialise the filter
    kf = KalmanFilter(X, P, F, Q, Z, H, R)
    lossAll=[]
    loss2=[]
    i=0
    output_data = codecs.open('./resdata4.txt','w','utf-8')
    
    for (x,y,axis) in zip(dataX,dataY,dataAxis):
       # for (x,y,axis) in zip(xr,yr,axisr):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)

        encoded,decoded,prediction = net(Variable(x))
        
        output_data.write(str(prediction.detach().numpy().tolist())+"/"+str(y.detach().numpy().tolist())+"\n")
        i=i+1 
        if i%200==0:
            print i
        #output_data.write('\n')
        '''
        for (index,Z) in enumerate(prediction.detach().numpy()):
            w = matrix(random.multivariate_normal([0.0, 0.0], Q)).T
            #w = 0
            (X, P) = kf.predict(X, P, w)
            (X, P) = kf.update(X, P, Z)
            kalman.append(X[0,0])
            
        loss1 = loss_func(prediction,y)
        lossAll.append(loss1.detach().numpy())
        
        kalman = torch.FloatTensor(kalman)
        loss1 = loss_func(kalman,y)
        loss2.append(loss1.detach().numpy())
        '''
            
        '''    
            ,prediction,y,kalman
            res = 0
            for x in lossAll:
                res = res+x
            print 'train',len(lossAll),pow(res/len(lossAll),0.5)
            res = 0
            for x in loss2:
                res = res+x
            print 'train',len(loss2),pow(res/len(loss2),0.5)
         '''      
            
    sdf        
    res = 0
    for x in lossAll:
        for y in x:
            res = res+pow(y,2)
    print 'test',len(lossAll),len(lossAll[0]),pow(res/(len(lossAll)*len(lossAll[0])),0.5)
    res = 0
    for x in loss2:
        for y in x:
            res = res+pow(y,2)
    print 'test',len(loss2),len(loss2[0]),pow(res/(len(loss2)*len(loss2[0])),0.5)
    '''
    plt.scatter(X, Y,s=10)
    plt.scatter(X, Z,s=10)
    plt.plot(X,E)
    plt.show()
    '''
testMSE(testdataX,testdataY,testdataAxis)   

