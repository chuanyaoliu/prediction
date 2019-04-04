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
from model import Net 


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
print len(data),len(data[0]),maxNum[5],minNum[5]
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

dataX,dataY = makeUpSeq(data,maxNum,minNum,sequence_len)

print len(dataX),len(dataY),len(dataX[0])
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
net = torch.load('./model/lstm_loss990.pkl')
#net = torch.load('./model/lstm_epoch1350.pkl')
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)  
loss_func = torch.nn.MSELoss() 

def test():
    X = []
    Y = []
    Z = []
    i=0
    #pre =torch.FloatTensor([x[4] for x in dataX[-51]])
    last = torch.FloatTensor(dataX[0])
    for (x,y) in zip(dataX[1:],dataY[1:]):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        encoded,decoded,prediction = net(Variable(last),True) 
        last = x
        i=i+1 
        X.append(i)
        Y.append(prediction[-1].detach().numpy()*0.5387+0.516)    
        Z.append(y[-1].detach().numpy()*0.5387+0.516)
    plt.scatter(X, Y, s=15)
    #plt.scatter(X, Z, s=15)
    plt.plot(X, Z)
    plt.show()
test()

for ep in range(1000):
    print ep
    i = 0
    res = 0
    res2 = 0
    X = []
    Y = []
    Z = []
    #last = torch.FloatTensor([x[4] for x in dataX[0]])
    last = torch.FloatTensor(dataX[0])
    for (x,y) in zip(dataX[1:250],dataY[1:250]):
       # x = torch.FloatTensor([xx[4] for xx in x]).view(30,-1)
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        encoded,decoded,prediction = net(Variable(last),True) 
        last = x
        
        i=i+1 
        X.append(i)
        Y.append(prediction[-1].detach().numpy()*0.5387+0.516)    
        Z.append(y[-1].detach().numpy()*0.5387+0.516)
        loss1 = loss_func(prediction[-1],y[-1])
        loss2 = loss_func(decoded[-1],x[-1])
        loss = loss1+loss2
        optimizer.zero_grad()  
        #loss.backward()        
        #optimizer.step()
               
        res+=loss1.detach().numpy() 
        res2+=loss2.detach().numpy()
        
        
    print 'train',pow(res/i,0.5),pow(res2/i,0.5),loss1.detach().numpy(),loss2.detach().numpy() 

    if ep!=0 and ep%30==0:
        model_name = "./model/lstm_loss"+str(ep)+".pkl"
        torch.save(net, model_name)
        print model_name,"has been saved"
        #testMSE(testdataX,testdataY,testdataAxis)
    
    plt.scatter(X, Y, s=15)
    #plt.scatter(X, Z, s=15)
    plt.plot(X, Z)
    plt.show()
    











