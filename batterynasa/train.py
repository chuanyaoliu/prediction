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
path = './nasa/B0005.mat'
cap,T1,X = readmat(path)
print len(X),len(cap),len(T1)
maxcap = max(cap)
cap = [float(x)/maxcap for x in cap]
maxT1 = max(T1)
T1 = [float(x)/maxT1 for x in T1]
'''
plt.plot(X,cap)
plt.show()
'''

sequence_len = 30
batch_size = 1
data = makeUpNasa(cap,T1,sequence_len)
print len(data)
'''
data,maxNum,minNum = readData('cs34.txt')
print len(data),len(data[0]),maxNum[5],minNum[5]

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
dataX,dataY = makeUpSeq(data,maxNum,minNum,sequence_len)

print len(dataX),len(dataY),len(dataX[0])

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

#net = Net(2, 50, sequence_len,batch_size)
net = torch.load('./model/lstm_loss10.pkl')
#net = torch.load('./model/lstm_epoch1350.pkl')
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  
loss_func = torch.nn.MSELoss() 

cricle = 70
def test():
    a = -0.000138
    b = 0.00468
    c=1.8623
    d=-0.0033
    n = 0.9952
    b1 = 0.1641
    b2=1.6023
    X = []
    Y = []
    Z = []
    Y1 = []
    Y2 =[]
    i=1
    res=0
    last = torch.FloatTensor(data[0])
    for x in data[1:]:
        prediction = net(last,True) 
        if i<cricle+200:
            #ck = x*maxcap
           # t = y*maxT1/24
            last = torch.FloatTensor(x)
            X.append(i)
            Y.append(prediction.detach().numpy()[-1])    
           # Y1.append(x*maxcap)  
           # Y2.append(a*math.exp(b*i)+c*math.exp(d*i))  
            Z.append(x[-1][0])
        else:
            #if y!=0:
            #    pre = n*ck+b1*math.exp(-b2/t)
            #else:
            #    pre = n*ck
            #ck = pre
            #t = y*maxT1/24
            #pre2 = a*math.exp(b*i)+c*math.exp(d*i)
            last.pop(0)
            last.push(prediction.detach().numpy())
            last = torch.FloatTensor([prediction.detach().numpy()[0],y]) 
            X.append(i)
            Y.append(prediction.detach().numpy()[0]*maxcap)   
            #Y1.append(pre) 
            #Y2.append(pre2)
            Z.append(x*maxcap)
            loss = loss_func(prediction*maxcap,ca*maxcap)
            res+=loss.detach().numpy() 
        i+=1
    print 'test',pow(res/(i-cricle),1)
        
    #plt.scatter(X, Y, s=15)
    #plt.scatter(X, Z, s=15)
    plt.plot(X,Y)
    #plt.plot(X,Y1)
    #plt.plot(X,Y2)
    plt.plot(X, Z)
    plt.show()
test()

for ep in range(1000):
    print ep
    i = 0
    res = 0
    X = []
    Y=[]
    Z=[]
    last = torch.FloatTensor(data[0])
    for x in data[1:40]:
        prediction = net(Variable(last),True) 
        last = torch.FloatTensor(x)
        y = [xx[0] for xx in x]
        y = torch.FloatTensor(y)
        i=i+1 
        loss = loss_func(prediction[-1],y[-1])
        optimizer.zero_grad()  
        loss.backward()        
        optimizer.step()
        X.append(i)
        Y.append(prediction.detach().numpy()[-1])
        Z.append(y.detach().numpy()[-1])
        res+=loss.detach().numpy() 
    print 'train',pow(res/i,0.5)
    
    if ep!=0 and ep%10==0:
        model_name = "./model/lstm_loss"+str(ep)+".pkl"
        torch.save(net, model_name)
        print model_name,"has been saved"
        #testMSE(testdataX,testdataY,testdataAxis)
        plt.plot(X,Y)
        plt.plot(X,Z)
        plt.show()
        
    











