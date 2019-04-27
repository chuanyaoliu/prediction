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
import torch.nn as nn
from torch.autograd import Variable
from util import  *
from model2 import Net 

path = './nasa/B0006.mat'
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

#net = Net(12, 50, sequence_len,batch_size)
#net = torch.load('./model/lstm_loss990.pkl')
net = torch.load('./model6/60annB0006.pkl')
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  
loss_func = torch.nn.MSELoss() 
'''
D = nn.Sequential(                  # Discriminator
    nn.Linear(2, 32), 
    nn.ReLU(),
    nn.Linear(32, 2),
    nn.ReLU()               
)
'''
D = torch.load('./model6/ganAD9950.pkl')
predict = torch.load('./model6/ganApre9950.pkl')
opt_D = torch.optim.SGD(D.parameters(), lr=0.1)
def softmax(x,y):
    temp = abs(x)+abs(y)
    return abs(y)/temp,abs(x)/temp
    

cricle = 60
if cricle ==60:
    a = 0.1477
    b = -0.0019
    c = 0.8919
    d = -0.0049
if cricle ==70:
    a = 0.1077
    b = -0.0019
    c = 0.8699
    d = -0.0039
n = 0.9928
b1 = 0.0787
b2=0.0759
def test():
    X = []
    X2=[]
    Y = []
    Z = []
    Y1 = []
    Y2 =[]
    i=1
    res=0
    last = torch.FloatTensor([cap[0],T1[0]])
    last2 = torch.FloatTensor([cap[0],T1[0]])
    d_input = []
    d_input2 = []
    for (x,y) in zip(cap[1:],T1[1:]):
        prediction = net(last,True) 
        prediction2 = net(last2,True) 
        pre2 = a*math.exp(b*i)+c*math.exp(d*i)
        '''
        inputdata = torch.FloatTensor([prediction,pre2])
        outputdata = D(inputdata)
        print outputdata
        outputdata = torch.mm(outputdata.view(1,-1),inputdata.view(-1,1)).view(-1)
        '''
        ca = torch.FloatTensor([x])
        if i<cricle:
            ck = x
            t = y*maxT1
            if t!=0:
                pre = n*ck+b1*math.exp(-b2/(t))
            else:
                pre = n*ck
            if len(d_input)>16:
                d_input.pop(0)
                d_input.pop(0)
                d_input.append(x)
                d_input.append(y)
            else:
                d_input.append(x)
                d_input.append(y)
            if len(d_input2)>16:
                d_input2.pop(0)
                d_input2.pop(0)
                d_input2.append(x)
                d_input2.append(y)
            else:
                d_input2.append(x)
                d_input2.append(y)
                
            last = torch.FloatTensor([x,y])
            last2 = torch.FloatTensor([x,y])
            X.append(i)
            
            Z.append(x*maxcap)
        else:
            
            if t!=0:
                pre = n*ck+b1*math.exp(-b2/(t))
            else:
                pre = n*ck
            ck = x
            t = y*maxT1
       
            if len(d_input)>16:
                d_input.pop(0)
                d_input.pop(0)
                d_input.append(prediction.detach().numpy()[0])
                d_input.append(y)
            else:
                d_input.append(prediction.detach().numpy()[0])
                d_input.append(y)
            if len(d_input2)>16:
                d_input2.pop(0)
                d_input2.pop(0)
                d_input2.append(pre2)
                d_input2.append(y)
            else:
                d_input2.append(pre2)
                d_input2.append(y)
                
            real = torch.FloatTensor([prediction,y])
            
            real_input = torch.cat((torch.FloatTensor(d_input),predict(real)))
            output2 = D(real_input)
            
            real = torch.FloatTensor([pre2,y])
            real = predict(real)
            real_input = torch.cat((torch.FloatTensor(d_input2),real))
            output1 = D(real_input)

            w1,w2 = softmax(output1.detach().numpy()[0],output2.detach().numpy()[0])
            fusion = prediction.detach().numpy()[0]*w1+pre2*w2
            print(w1,w2)
            last = torch.FloatTensor([prediction.detach().numpy()[0],y]) 
            last2 = torch.FloatTensor([fusion,y]) 
            X.append(i)
            X2.append(i)
            if x*maxcap<1.4:
                print(i)
            
            Y.append(prediction.detach().numpy()[0]*maxcap)   
            Y1.append(prediction2.detach().numpy()[0]*maxcap)
            Y2.append(pre2*maxcap)
            Z.append(x*maxcap)
            
            loss = loss_func( prediction*maxcap,ca*maxcap)
            res+=loss.detach().numpy() 
        i+=1
    print('test')
    print('test',pow(res/(i-cricle),1),pow(res/(i-cricle),0.5))

    #plt.scatter(X, Y, s=15)
    #plt.scatter(X, Y1, s=15)
    if cricle==70:
        plt.plot(X,Z, label='real capacity',color= '#FF7F00')
        plt.plot(X2,Y,label='ann model',color='#377EB8')
        plt.plot(X2,Y1,label='Fusion model', color= '#984EA3')
        plt.plot(X2,Y2,label='Empirical model',color='#4DAF4A')
        plt.legend(loc='upper right', fontsize=10);
        plt.hlines(1.4, 0,175 , color='darkgrey', linestyles = "dashed")
        plt.vlines(70, 1.2,2 , color='k')
        plt.vlines(118, 1.2,1.6 , color='#377EB8')  
        plt.text(118, 1.55, r'118', fontsize=10)
        plt.vlines(105, 1.2,1.6 , color='#984EA3')
        plt.text(95, 1.55, r'105', fontsize=10)
        plt.vlines(96, 1.2,1.6 , color='#4DAF4A')
        plt.text(92, 1.62, r'96', fontsize=10)
        plt.vlines(108, 1.2,1.6 , color='#FF7F00')
        plt.text(105, 1.62, r'108', fontsize=10)
        #plt.xlim(0,175)
        #plt.ylim(1.2,2)
        plt.show()
    elif cricle==60:
        plt.plot(X,Z, label='real capacity',color= '#FF7F00')
        plt.plot(X2,Y,label='ann model',color='#377EB8')
        plt.plot(X2,Y1,label='Fusion model', color= '#984EA3')
        plt.plot(X2,Y2,label='Empirical model',color='#4DAF4A')
        plt.legend(loc='upper right', fontsize=10);
        plt.hlines(1.4, 0,175 , color='darkgrey', linestyles = "dashed")
        plt.vlines(60, 1.2,2 , color='k')
        #plt.vlines(140, 1.2,1.6 , color='#377EB8')  
        #plt.text(140, 1.55, r'140', fontsize=10)
        plt.vlines(105, 1.2,1.6 , color='#984EA3')
        plt.text(95, 1.55, r'105', fontsize=10)
        plt.vlines(94, 1.2,1.6 , color='#4DAF4A')
        plt.text(92, 1.62, r'95', fontsize=10)
        plt.vlines(108, 1.2,1.6 , color='#FF7F00')
        plt.text(105, 1.62, r'108', fontsize=10)
        #plt.xlim(0,175)
        #plt.ylim(1.2,2)
        plt.show()
    return None
test()

sss
for ep in range(10000):
    print(ep)
    i = 1
    res = 0
    X = []
    Y = []
    Y1 = []
    Y2 = []
    Z = []
    last = torch.FloatTensor([cap[0],T1[0]])
    for (x,y) in zip(cap[1:cricle],T1[1:cricle]):
        prediction = net(last,True) 
        ca = torch.FloatTensor([x])
        last = torch.FloatTensor([x,y])
        pre = a*math.exp(b*i)+c*math.exp(d*i)
        inputdata = torch.FloatTensor([prediction,pre])
        outputdata1 = D(inputdata)
        outputdata = torch.mm(outputdata1.view(1,-1),inputdata.view(-1,1)).view(-1)
        X.append(i)
        Y.append(prediction.detach().numpy()[0])    
        Y1.append(pre)  
        Y2.append(outputdata.detach().numpy()[0]) 
        Z.append(x)

        i=i+1 
        
        loss = loss_func(outputdata,ca)
        opt_D.zero_grad()  
        loss.backward()        
        opt_D.step()
        res+=loss.detach().numpy() 
        
    print ('train',pow(res/i,1),outputdata1)

    if ep!=0 and ep%200==0:
        model_name = "./model/model_bagging"+str(ep)+".pkl"
        torch.save(D, model_name)
        print( model_name,"has been saved"  )
        
        plt.scatter(X, Y, s=15)
        #plt.scatter(X, Z, s=15)
        plt.plot(X,Y1)
        plt.scatter(X,Y2)
        plt.plot(X, Z)
        plt.show()
        test()
        










