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
import torch.nn.functional as F     # 激励函数都在这
from torch.autograd import Variable
from util import * 
from encodermodel import AutoEncoder


sequence_len = 30
maxNum,minNum = findMaxMin('CMAPSSData/train_FD001.txt')

data = readFile('CMAPSSData/train_FD001.txt')
testrul = readrul('CMAPSSData/RUL_FD001.txt')
data = selectFeature(data,maxNum,minNum)
dataX,dataY,dataAxis = makeUpEncoder(data,sequence_len)


            

#autoencoder = AutoEncoder()
autoencoder = torch.load('./model/encoder_epoch120.pkl')
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0001)
loss_func = nn.MSELoss()



X,Y,Z = encoderReturn(autoencoder,dataX,dataY,dataAxis)
print len(X)
for i in range(len(X)):
    plt.plot(Z[i],X[i])
#plt.scatter(X, Y, s=10)
plt.show()
    
for ep in range(500):
    lossAll=0
    X=[]
    Y=[]
    Z=[]
    for (x,y,axis) in zip(dataX,dataY,dataAxis):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        encoded, decoded = autoencoder(Variable(x))
        loss = loss_func(decoded, x)     
        optimizer.zero_grad()               
        loss.backward()                    
        optimizer.step()  
        lossAll = lossAll+loss
        #Y1= flowList(encoded.detach().numpy())
        X.append(axis)
        Y.append(y.detach().numpy())
        Z.append(encoded.detach().numpy())
        
    print lossAll,len(X),len(Y),len(Z),len(X[0])
    
    if ep%30==0:
        model_name = "./model/encoder_epoch"+str(ep)+".pkl"
        torch.save(autoencoder, model_name)
        print model_name,"has been saved"
        
        for i in range(len(X)):
            plt.plot(X[i],Z[i])
        #plt.scatter(X, Y, s=10)
        plt.show()
        test()  
