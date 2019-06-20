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
from cnn import Net 



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
 
''' 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
 
 
ax.scatter(-0.8872,  0.5771,  0.5440)
ax.scatter(  0.1473,  1.1259, -0.2291)
ax.scatter(-0.6157,  0.4774, -1.1102)
ax.scatter(  0.4373, -0.0682,  1.5442)
ax.scatter(1.3546,  0.1592, -0.4990)
ax.scatter(-0.2813, -1.3465, -0.1079)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
 
plt.show()

'''

filename = 'FD002'
sequence_len=30
batch_size = 1
'''
a = [0,1,0,0,0,0,0.3,0.2,0.5,0,0,0.6,0.1,0.1,0,0.4,0.5,0.8,0.2,0.3,0,0.4,0,0,0.9,0.7]
a = np.asarray(a)
print [a[i] for i in[6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,24,25]]
'''
maxNum,minNum = findMaxMin2('CMAPSSData/train_%s.txt' % filename)
#print maxNum,minNum
data = readFile('CMAPSSData/train_%s.txt'% filename)
'''
for jj in range(16):
    X=[]
    Y=[]
    for x in range(len(data)):
        for i in range(len(data[x])):
            X.append(data[x][i][1])
            Y.append(data[x][i][jj])
    plt.scatter(X, Y, s=15)
    #plt.scatter(X, Z, s=15)
    #plt.xlim(-10, 300)
    #plt.ylim(-10, 300)
    plt.xlabel('Cycle')
    plt.tick_params(direction='in')
    plt.savefig('560.png', dpi=800)
    plt.show()
'''
data,condition = selectFeature2(data,maxNum,minNum)
print len(data),len(data[0]),len(condition),len(condition[0])
'''
for jj in range(26):
    X=[]
    Y=[]
    for x in range(len(data)):
        meanNum = np.mean(data[x],axis=0)
        stdNum = np.std(data[x],axis=0)
        for i in range(len(data[x])):
            X.append(data[x][i][0])
            Y.append(data[x][i][jj])
        
    plt.scatter(X, Y, s=15)
    #plt.scatter(X, Z, s=15)
    #plt.xlim(-10, 300)
    #plt.ylim(-10, 300)
    plt.xlabel('Cycle')
    plt.tick_params(direction='in')
    plt.savefig('560.png', dpi=800)
    plt.show()
'''

dataX,dataY,dataAxis = makeUpSeq2(data,sequence_len)
data = None

testdata = readFile('CMAPSSData/test_%s.txt'% filename)
testdata,conditiontest = selectFeature2(testdata,maxNum,minNum)
testrul = readrul('CMAPSSData/RUL_%s.txt'% filename)
testdataX,testdataY,testdataAxis = makeUpSeq2(testdata,sequence_len,testrul)
testdata = None
#net = Net(17,3, 300, sequence_len,batch_size)
net = torch.load('./model/a_data2_multi3.pkl')
#net = torch.load('./model/data4_epoch201.pkl')
#print net.condition_embeds(torch.LongTensor([0,1,2,3,4,5]))

#net = torch.load('./modelNew/model3_epoch18.pkl')
learning_rate = 0.00001
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)  
'''
conv5_params = list(map(id, net.encoder.parameters()))
base_params = filter(lambda p: id(p) not in conv5_params,
                     net.parameters())
conv5_params = list(map(id, net.decoder.parameters()))
base_params = filter(lambda p: id(p) not in conv5_params,
                     base_params)
optimizer = torch.optim.SGD([
            {'params': base_params},
            {'params': net.encoder.parameters(), 'lr': 0.000001},
            {'params': net.decoder.parameters(), 'lr': 0.000001}],
            lr=0.00001)  
'''           
loss_func = torch.nn.MSELoss()   

def testMSE(dataX,dataY,dataAxis,condition):
    i=0
    res=0
    score=0
    X=[]
    Y=[]
    Z=[]
    for (x,y,axis,c) in zip(dataX,dataY,dataAxis,condition):
        x = torch.FloatTensor(x)
        c = torch.LongTensor(c)
        prediction = net(Variable(x),Variable(c),False)
        loss = loss_func(prediction[-1],torch.FloatTensor(y)[-1])
        i+=1
        res+= loss.detach().numpy() 
        score+= calScore(prediction.detach().numpy().tolist()[-1],y[-1])
        '''
        if i%2000==0:
            print i,prediction,y
            print 'test',pow(res/i,0.5)
        '''
        X=[]
        print i
        if i in (45,70,76,103,116,138,158,159,185):
            for xx in range(len(prediction)):
                X.append(xx)
            plt.scatter(X, prediction.detach().numpy().tolist(),s=15,label="Predicted RUL")
            plt.plot(X, y[18:],label="Actual RUL")
            plt.legend(loc='upper right', fontsize=10);
            plt.tick_params(direction='in')
            plt.xlabel('Time Cycle')
            plt.ylabel('RUL')
            plt.savefig('570.png', dpi=800)
            plt.show()
        #X.append(i)
        #Y.append(prediction.detach().numpy().tolist()[-1])
        #Z.append(y[-1])
    
    def sortAll(pre,true):
        length = len(true)
        for i in range(1,length):
            for j in range(0,i):
                if true[i-j]<true[i-j-1]:
                    temp = true[i-j]
                    true[i-j] = true[i-j-1]
                    true[i-j-1] = temp
                    temp = pre[i-j]
                    pre[i-j] = pre[i-j-1]
                    pre[i-j-1] = temp
    #sortAll(Y,Z)
    #plt.plot(X, Y)
    #plt.plot(X, Z)
    #plt.plot(X,E)
    #plt.tick_params(direction='in')
    #plt.xlabel('Cycle')
    #plt.ylabel('Capacity/Ahr')
    #plt.savefig('570.jpg', dpi=300)
    print 'test',pow(res/i,0.5),score
    #plt.show()
    
testMSE(testdataX,testdataY,testdataAxis,conditiontest)   
for ep in range(40000):
    i = 0
    res = 0
    for (x,y,axis,c) in zip(dataX,dataY,dataAxis,condition):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        c = torch.LongTensor(c)
        prediction = net(Variable(x),Variable(c),True) 
        #print net.condition_embeds(c)
        loss1 = loss_func(prediction,y[18:])
        #print loss1,loss2
        optimizer.zero_grad()  
        net.zero_grad()
        loss1.backward()        
        optimizer.step()
        i=i+1 
        res+=loss1.detach().numpy() 
        if i%5000==0:
            print i,loss1,prediction,y
            print 'train',pow(res/i,0.5)
    
    
    if ep!=0 and ep%50==0:
        print ep,loss1
        #if ep%1000==0:
        #    learning_rate /= 10        
        print 'train',pow(res/i,0.5)
        model_name = "./model/data4_epoch"+str(ep+1)+".pkl"
        torch.save(net, model_name)
        print model_name,"has been saved"
        testMSE(testdataX,testdataY,testdataAxis,conditiontest)

        
    #plt.scatter(X, Y,s=10)
    #plt.scatter(X, Z,s=10)
    #plt.plot(X,E)
    #plt.show()
    
    

