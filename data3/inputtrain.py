#coding:utf8
import codecs
import pdb
import sys
import math
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

from kalman_model import Kalman

sequence_len=30
batch_size = 1

#maxNum,minNum = findMaxMin('CMAPSSData/train_FD001.txt')

#data = readFile('CMAPSSData/train_FD001.txt')
#data = selectFeature(data,maxNum,minNum)
#dataX,dataY,dataAxis = makeUpSeq(data,sequence_len)

#testdata = readFile('CMAPSSData/test_FD001.txt')
#testdata = selectFeature(testdata,maxNum,minNum)
#testrul = readrul('CMAPSSData/RUL_FD001.txt')
#testdataX,testdataY,testdataAxis = makeUpSeq(testdata,sequence_len,testrul)

#net = Net(1, 300, sequence_len,batch_size)
#net = torch.load('./model/model_epoch30.pkl')
#net = torch.load('./modelNew/model_epoch29.pkl')
#net = torch.load('./modelNew/model3_epoch18.pkl')
#optimizer = torch.optim.SGD(net.parameters(), lr=0.001)  
'''
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
'''           
def calScore(pre,y):
    res=0
    #for pr,yr in zip(pre,y):
    d = pre-y
    if d<0:
        res+=math.exp(-d/13)
    else:
        res+=math.exp(d/10)   
        #res+=abs(d)
    return res#/len(y)
from kalman_filter import KalmanFilter
from numpy import matrix, diag, random



dt = 0.04
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
R = matrix([[sigma_z ** 2, 0], [0, 0.001]])
# Process noise covariance matrix
Q = G * (G.T) * sigma_a ** 2
#Q = matrix([[0.00001, 0.001], [0, 0.01]])

# Initialise the filter
kf = KalmanFilter(X, P, F, Q, Z, H, R)


#F = [[1, 1], [0, 1]]
#P =torch.FloatTensor([[0,0],[0,0]])
#H = [[1, 0], [0, 0]]

#kf = Kalman(F,H)

#optimizer = torch.optim.SGD(kf.parameters(),lr=0.00001)  
loss_func = torch.nn.MSELoss()  


lossAll=[]
loss2=[]
i=0
input_data = codecs.open('./resdata2.txt','r','utf-8')
for line in input_data.readlines():
    line=line.strip()
    line = line.split('/')
    pre = line[0][1:-1].split(',')
    pre = map(eval,pre)
    y = line[1][1:-1].split(',')
    y = map(eval,y)
    
    kalman = []

   # X = torch.FloatTensor([[pre[0]], [0.0]])
   # P = torch.FloatTensor([[0,0],[0,0]])
    X =[[pre[0]], [0.0]]
    #P =[[0,0],[0,0]]
    for jj in range(1):
        ax = []
        for (index,(Z,yr)) in enumerate(zip(pre,y)):
            w = matrix(random.multivariate_normal([0.0, 0.0], Q)).T
            #w=0
            (X, P) = kf.predict(X, P, w)
            (X, P) = kf.update(X, P, Z)
            if jj==0:
                kalman.append(X[0,0])
                ax.append(index)
            '''
            Z = torch.FloatTensor([Z])
            (X, P) = kf(X, P, Z)     
            yr = torch.FloatTensor([yr])
            loss = loss_func(X[0,0],Z)
            optimizer.zero_grad()  
            loss.backward(retain_graph=True)        
            optimizer.step()
            if jj==0:
                kalman.append(X[0,0].detach().numpy().tolist())
            '''
    
    plt.plot(ax, kalman)
    plt.plot(ax, y)
    plt.scatter(ax, pre, s=15)
    plt.legend('kyl')
    
    #plt.xlim(-10, 300)
    #plt.ylim(-10, 300)
    plt.show()
    
    loss = loss_func(torch.FloatTensor(pre),torch.FloatTensor(y))
    #lossAll.append(loss.detach().numpy())
    #lossAll.append(calScore(pre[-1],y[-1]))
    loss1 = loss_func(torch.FloatTensor(kalman),torch.FloatTensor(y))
    #loss2.append(loss1.detach().numpy())
    #loss2.append(calScore(kalman[-1],y[-1]))
    lossAll.append(abs(y[-1]-pre[-1]))
    loss2.append(abs(y[-1]-kalman[-1]))
    i=i+1 
    if i%200==0:
        print i,loss,loss1,kf.Q,kf.R,P
        
        
           
             
res = 0
for x in lossAll:
    res = res+pow(x,2)
print 'train',len(lossAll),pow(res/len(lossAll),0.5),res/len(lossAll)
res = 0
for x in loss2:
    res = res+pow(x,2)
print 'train',len(loss2),pow(res/len(loss2),0.5),res/len(lossAll)



