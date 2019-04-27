#coding:utf8
import codecs
import pdb
import sys
import copy
import pandas as pd
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F     
from torch.autograd import Variable
from util import  *
from model2 import Net 

path = './nasa/B0005.mat'
cap,T1,X = readmat(path)
print (len(X),len(cap),len(T1))
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
'''
a=[[1,2,3,4],[1,2,3,4],[1,2,3,4]]
a = np.asarray(a)
print a[:,(1,3)]

path = "./CS2_34/" 
data = readXLSX(path)
writeData(data,'cs34.txt')

sequence_len = 30
batch_size = 1
data,maxNum,minNum = readData('cs34.txt')
print len(data),len(data[0]),maxNum[1],minNum[1]

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


dataX,dataY = makeUpSeq3(data,maxNum,minNum,sequence_len)

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
learning_rate = 0.0001

#net = Net(12, 50, sequence_len,batch_size)
#net = torch.load('./model/lstm_loss990.pkl')
#net = torch.load('./model/07ann.pkl')
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)  
loss_func = torch.nn.MSELoss() 

cricle = 70
def test():
    a = 0.0027
    b = -0.1983
    c = 0.9975
    d = -0.0021
    n = 0.9928
    b1 = 0.0787
    b2=0.0759
    X = []
    Y = []
    Z = []
    Y1 = []
    Y2 =[]
    i=1
    res=0
    last = torch.FloatTensor([cap[0],T1[0]])
    last2 = torch.FloatTensor([cap[0],T1[0]])
    for (x,y) in zip(cap[1:],T1[1:]):
        prediction = net(last,True) 
       # prediction2 = net(last2,True) 
        ca = torch.FloatTensor([x])
        if i<cricle:
            ck = x
            t = y*maxT1
            if t!=0:
                pre = n*ck+b1*math.exp(-b2/(t))
            else:
                pre = n*ck
            last = torch.FloatTensor([x,y])
            last2 = torch.FloatTensor([x,y])
            X.append(i)
            Y.append(prediction.detach().numpy()[0]*maxcap)    
            Y1.append(pre*maxcap)  
            Y2.append((a*math.exp(b*i)+c*math.exp(d*i))*maxcap) 
            Z.append(x*maxcap)
        else:
            if t!=0:
                pre = n*ck+b1*math.exp(-b2/(t))
            else:
                pre = n*ck
            ck = x
            t = y*maxT1
            pre2 = a*math.exp(b*i)+c*math.exp(d*i)
            last = torch.FloatTensor([prediction.detach().numpy()[0],y]) 
            #last2 = torch.FloatTensor([prediction2[0],y]) 
            X.append(i)
            Y.append(prediction.detach().numpy()[0]*maxcap)   
            Y1.append(pre*maxcap) 
            Y2.append(pre2*maxcap)
            Z.append(x*maxcap)
            loss = loss_func(prediction*maxcap,ca*maxcap)
            res+=loss.detach().numpy() 
        i+=1
    print('test',pow(res/(i-cricle),1))
        
    #plt.scatter(X, Y, s=15)
    #plt.scatter(X, Z, s=15)
    plt.plot(X,Y)
    #plt.plot(X,Y1)
    plt.plot(X,Y2)
    plt.plot(X, Z)
    plt.show()
#test()

for ep in range(10000):
    print(ep)
    i = 1
    res = 0
    X = []
    Y = []
    Z = []
    last = torch.FloatTensor([cap[0],T1[0]])
    for (x,y) in zip(cap[1:cricle],T1[1:cricle]):
        prediction = net(last,True) 
        ca = torch.FloatTensor([x])
        
        #last = torch.FloatTensor([prediction.detach().numpy()[0],y]) 
        i=i+1 
        X.append(i)
        Y.append(prediction.detach().numpy()[0])    
        Z.append(x)

        loss = loss_func(prediction,ca)
        last = torch.FloatTensor([x,y])
        optimizer.zero_grad()  
        loss.backward()        
        optimizer.step()
        res+=loss.detach().numpy() 
        
    print ('train',pow(res/i,1))
    if ep!=0 and ep%1000==0:
        learning_rate /= 10
    if ep!=0 and ep%500==0:
        model_name = "./model/model_epoch"+str(ep)+".pkl"
        torch.save(net, model_name)
        print( model_name,"has been saved"  )
        
        plt.scatter(X, Y, s=15)
        #plt.scatter(X, Z, s=15)
        #plt.plot(X,Y)
        plt.plot(X, Z)
        plt.show()
        test()
        










