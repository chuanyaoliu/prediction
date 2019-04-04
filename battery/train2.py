#coding:utf8
import codecs
import pdb
import sys
import copy
import pandas as pd
import xlrd
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F     
from torch.autograd import Variable
from util import  *
from model2 import Net 


'''
a=[[1,2,3,4],[1,2,3,4],[1,2,3,4]]
a = np.asarray(a)
print a[:,(1,3)]

path = "./CS2_34/" 
data = readXLSX(path)
writeData(data,'cs34.txt')
'''
sequence_len = 30
batch_size = 1
data,maxNum,minNum = readData('cs34.txt')
print len(data),len(data[0]),maxNum[6],minNum[6]
'''
for j in range(len(data[0])):
    X = []
    Y = []
    i=1
    for x in data:
        X.append(i)
        Y.append(x[j])
        i+=1
    #plt.scatter(X, Y, s=15)
    plt.plot(X,Y)
    plt.show()    
'''

#dataX,dataY = makeUpSeq(data,maxNum,minNum,sequence_len)

#print len(dataX),len(dataY),len(dataX[0])
'''
for j in range(len(dataX)):
    X = []
    Y = []
    i=1
    for (x,y) in zip(dataX,dataY):
        X.append(i)
        Y.append(x[-1][j])
        i+=1
    plt.scatter(X, Y, s=15)
    plt.show()    
'''

#net = Net(12, 100, sequence_len,batch_size)
#net = torch.load('./model/lstm_loss990.pkl')
net = torch.load('./model/model_epoch210.pkl')
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)  
loss_func = torch.nn.MSELoss() 

def test():
    X = []
    Y = []
    Z = []
    i=0
    last = data[0]
    for line in data[1:]:
        n = last[5]*last[5]/last[4]
        x = torch.FloatTensor(i)
        n = torch.FloatTensor([n])
        if last[1]!=0:
            t = torch.FloatTensor([last[1]])
        else:
            t = 1
        y = torch.FloatTensor([line[5]])
        prediction = net(x,n,t,True) 
        last = line
        print prediction.detach().numpy()
        last[5] = prediction.detach().numpy()[0]
        i=i+1 
        X.append(i)
        Y.append(prediction.detach().numpy()*0.5387+0.516)    
        Z.append(y.detach().numpy()*0.5387+0.516)
    plt.scatter(X, Y, s=15)
    #plt.scatter(X, Z, s=15)
    plt.plot(X, Z)
    plt.show()
#test()

for ep in range(1000):
    print ep
    i = 0
    res = 0
    res2 = 0
    X = []
    Y = []
    Z = []
    #last = torch.FloatTensor([x[4] for x in dataX[0]])
    last = data[0]
    for line in data[1:250]:
        n = last[5]*last[5]/last[4]
        x = torch.FloatTensor(i)
        n = torch.FloatTensor([n])
        if last[1]!=0:
            t = torch.FloatTensor([last[1]])
        else:
            t = 1
        y = torch.FloatTensor([line[5]])
        prediction = net(x,n,t,True) 
        #print last[5],prediction,line[5]
        last = line
        i=i+1 
        X.append(i)
        Y.append(prediction.detach().numpy())    
        Z.append(y.detach().numpy())
        loss = loss_func(prediction,y)
        optimizer.zero_grad()  
        loss.backward()        
        optimizer.step()
               
        res+=loss.detach().numpy() 
        
        
    print 'train',pow(res/i,0.5),net.b1,net.b2,net.w

    if ep!=0 and ep%30==0:
        model_name = "./model/model_epoch"+str(ep)+".pkl"
        torch.save(net, model_name)
        print model_name,"has been saved"
        #testMSE(testdataX,testdataY,testdataAxis)
   
        plt.scatter(X, Y, s=15)
        #plt.scatter(X, Z, s=15)
        plt.plot(X, Z)
        plt.show()












