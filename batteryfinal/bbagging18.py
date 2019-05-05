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

path = './nasa/B0018.mat'
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
net = torch.load('./model18/60ann4000.pkl')
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
D = torch.load('./model18/ganAD9950.pkl')
predict = torch.load('./model18/ganApre9950.pkl')
opt_D = torch.optim.SGD(D.parameters(), lr=0.1)
   

cricle = 60
if cricle ==60:
    a = 0.3978
    b = -0.0019
    c = 0.6952
    d = -0.0056
if cricle ==70:
    a = 0.3978
    b = -0.0019
    c = 0.6952
    d = -0.0056
n = 0.9959
b1 = 0.1641
b2=0.9023
def test():
    X = []
    X2=[]
    Y = []
    Z = []
    Y1 = []
    Y2 =[]
    Y3=[]
    i=1
    res=0
    last = torch.FloatTensor([cap[0],T1[0]])
    last2 = torch.FloatTensor([cap[0],T1[0]])
    d_input = []
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
            ck = x*maxcap
            t = y*maxT1/float(24)
            if len(d_input)>16:
                d_input.pop(0)
                d_input.pop(0)
                d_input.append(x)
                d_input.append(y)
            else:
                d_input.append(x)
                d_input.append(y)
            
            last = torch.FloatTensor([x,y])
            last2 = torch.FloatTensor([x,y])
            X.append(i)
            
            Z.append(x*maxcap)
        else:
            
            if t!=0:
                pre = n*ck+b1*math.exp(-b2/(t))
            else:
                pre = n*ck
            ck = pre
            t = y*maxT1/float(24)
       
            if len(d_input)>16:
                d_input.pop(0)
                d_input.pop(0)
                d_input.append(prediction.detach().numpy()[0])
                d_input.append(y)
            else:
                d_input.append(prediction.detach().numpy()[0])
                d_input.append(y)
                
            real = torch.FloatTensor([prediction,y])
            
            real_input = torch.cat((torch.FloatTensor(d_input),predict(real)))
            output1 = D(real_input)
            
                
            if len(d_input)>16:
                d_input.pop(0)
                d_input.pop(0)
                d_input.append(pre2)
                d_input.append(y)
            else:
                d_input.append(pre2)
                d_input.append(y)
                
            
            real = torch.FloatTensor([pre2,y])
            real = predict(real)
            real_input = torch.cat((torch.FloatTensor(d_input),real))
            output2 = D(real_input)
            
            
            if len(d_input)>16:
                d_input.pop(0)
                d_input.pop(0)
                d_input.append(pre)
                d_input.append(y)
            else:
                d_input.append(pre)
                d_input.append(y)
                
            
            real = torch.FloatTensor([pre/maxcap,y])
            real = predict(real)
            real_input = torch.cat((torch.FloatTensor(d_input),real))
            output3 = D(real_input)
            
            
            
            w = softmax([output1.detach().numpy()[0],output2.detach().numpy()[0],output3.detach().numpy()[0]])
            print(w)
            fusion = prediction.detach().numpy()[0]*w[2]+pre2*w[1]+pre/maxcap*w[0]
            last = torch.FloatTensor([prediction.detach().numpy()[0],y]) 
            last2 = torch.FloatTensor([fusion,y]) 
            X.append(i)
            X2.append(i)
            if prediction.detach().numpy()*maxcap<1.4:
                print(i)
            
            Y.append(prediction.detach().numpy()[0]*maxcap)   
            Y1.append(fusion*maxcap)
            Y2.append(pre2*maxcap)
            Y3.append(pre)
            Z.append(x*maxcap)
            
            loss = loss_func( torch.FloatTensor([fusion])*maxcap,ca*maxcap)
            res+=loss.detach().numpy() 
        i+=1
    print('test')
    print('test',pow(res/(i-cricle),1))

    #plt.scatter(X, Y, s=15)
    #plt.scatter(X, Y1, s=15)
    plt.plot(X2,Y,label='ann model')
    plt.plot(X2,Y1,label='Fusion model')
    plt.plot(X2,Y2,label='Empirical model')
    plt.plot(X2,Y3, label='Empirical model2',color='darkgreen')
    plt.plot(X,Z)
    plt.legend(loc='upper right', fontsize=10);
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
        










