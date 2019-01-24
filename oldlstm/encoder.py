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


'''
b = np.array([[[2,3,5],
             [2,3,5],
             [2,3,5]],
             [[2,3,5],
             [2,3,5],
             [2,3,5]]])
a = np.array([1,2,3])
print a[[0,2]]
'''
sequence_len = 30


data = []
oneData=[]
run = 1
with open('CMAPSSData/test_FD001.txt') as inp:
    for line in inp.readlines():
        line = line.split()
        if int(line[0])==run:
            oneData.append(map(eval,line))
        else:
            data.append(oneData)
            oneData=[]
            oneData.append(map(eval,line))
            run=int(line[0])
            #if run==33:
            #    break
    data.append(oneData)

print 'train',len(data),len(data[0])

data = np.asarray(data)

for x in range(len(data)):
    maxNum = np.max(data[x],axis=0)
    minNum = np.min(data[x],axis=0)
    
    for i in range(len(maxNum)):
        if maxNum[i]==minNum[i]:
            maxNum[i]=maxNum[i]+1
    minNum[1]=0
    maxNum[1]=1
    data[x]= (data[x]-minNum)/(maxNum-minNum)
    data[x] = data[x][:,[1,2,3,6,7,8,11,12,13,15,16,17,18,19,21,24,25]]


                 


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(16, 200),
            nn.Tanh(),
            nn.Linear(200, 50),
            nn.Tanh(),
            nn.Linear(50, 1),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 200),
            nn.Tanh(),
            nn.Linear(200, 16),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

#autoencoder = AutoEncoder()
autoencoder = torch.load('./encoder2_epoch200.pkl')
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0001)
loss_func = nn.MSELoss()


def test():
    X=[]
    Y=[]
    Z=[]
    for i in range(len(data)):
        X1=[]
        Y1=[]
        for j in range(len(data[i])):
            x = data[i][j][1:]
            x = torch.FloatTensor(x)
            y = data[i][j][0]-len(data[i])
            encoded, decoded = autoencoder(Variable(x))
            X1.append(y)
            Y1.append(encoded.item())
        Y2=[]    
        for k in range(len(Y1)):
            if k<4 or k>len(Y1)-4:
                Y2.append(Y1[k])
            else:
                Y2.append((Y1[k-1]+Y1[k-2]+Y1[k-3]+Y1[k]+Y1[k+1]+Y1[k+2]+Y1[k+3])/7)
        for k in range(len(Y1)):
            if k<4 or k>len(Y1)-4:
                continue
            else:
                Y1[k]=(Y2[k-1]+Y2[k-2]+Y2[k-3]+Y2[k]+Y2[k+1]+Y2[k+2]+Y2[k+3])/7
        for k in range(len(Y1)):
            if k<4 or k>len(Y1)-4:
                continue
            else:
                Y2[k]=(Y1[k-1]+Y1[k-2]+Y1[k-3]+Y1[k]+Y1[k+1]+Y1[k+2]+Y1[k+3])/7
        X.append(X1)
        Y.append(Y2)
    
    for i in range(len(X)):
        plt.plot(X[i],Y[i])
    #plt.scatter(X, Y, s=10)
    plt.show()
test()

for ep in range(500):
    lossAll=0
    #X=[]
    #Y=[]
    #Z=[]
    for i in range(len(data)):
        #X1=[]
        #Y1=[]
        for j in range(len(data[i])):
            x = data[i][j][1:]
            x = torch.FloatTensor(x)
            y = data[i][j][0]-len(data[i])
            encoded, decoded = autoencoder(Variable(x))
            loss = loss_func(decoded, x)      # mean square error
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()  
            lossAll = lossAll+loss
            
            #X1.append(y)
            #Y1.append(encoded.item())
        #X.append(X1)
        #Y.append(Y1)
    print lossAll,encoded
    
    if ep%10==0:
        model_name = "./encoder2_epoch"+str(ep)+".pkl"
        torch.save(autoencoder, model_name)
        print model_name,"has been saved"
        '''
        for i in range(len(X)):
            plt.plot(X[i],Y[i])
        #plt.scatter(X, Y, s=10)
        plt.show()
        '''
