import copy
import numpy as np
import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

def findMaxMin(filename):
    data = []
    oneData=[]
    maxNum = [0 for i in range(26)]
    minNum = [100, 362, 0.0087, 0.0006, 100.0, 518.67, 644.53, 1616.91, 1441.49, 14.62, 21.61, 556.06, 2388.56, 9244.59, 1.3, 48.53, 523.38, 2388.56, 8293.72, 8.5848, 0.03, 400, 2388, 100.0, 39.43, 23.6184]
    with open(filename) as inp:
        for line in inp.readlines():
            line = line.split()
            line = map(eval,line)
            for i in range(26):
                if line[i]>maxNum[i]:
                    maxNum[i]=line[i]
                if line[i]<minNum[i]:
                    minNum[i]=line[i]
            
    return maxNum,minNum
def readFile(filename):
    data = []
    oneData=[]
    run = 1
    with open(filename) as inp:
        for line in inp.readlines():
            line = line.split()
            if int(line[0])==run:
                oneData.append(map(eval,line))
            else:
                data.append(oneData)
                oneData=[]
                oneData.append(map(eval,line))
                run=int(line[0])
                #if run==5:
                 #   break
        data.append(oneData)
    print filename,len(data),len(data[0])
    return data
    

def selectFeature(data,maxNum,minNum):
    data = np.asarray(data)
    maxNum = np.asarray(maxNum)
    minNum= np.asarray(minNum)
    for x in range(len(data)):
        for i in range(len(maxNum)):
            if maxNum[i]==minNum[i]:
                maxNum[i]=maxNum[i]+1
                
        minNum[1]=0
        maxNum[1]=1
        data[x]= (data[x]-minNum)/(maxNum-minNum)
        data[x] = data[x][:,[1,2,3,6,7,8,11,12,13,15,16,17,18,19,21,24,25]]

    return data

def makeUpSeq(data,sequence_len,rul=[]):
    dataX = []
    dataY = []
    dataAxis = []
    for i in range(len(data)):
        sequenceX = []
        sequenceY = []
        seqAxis = []
        for j in range(len(data[i])):
            x = data[i][j][1:]
            if len(rul)!=0:
                y = len(data[i])-data[i][j][0]+rul[i]
            else:
                y = len(data[i])-data[i][j][0]
            seqAxis.append(y)  
            y = y if y<130 else 130
            sequenceX.append(x)
            sequenceY.append(y)
            if j>sequence_len-2:
                dataX.append(copy.deepcopy(sequenceX))
                dataY.append(copy.deepcopy(sequenceY))
                dataAxis.append(copy.deepcopy(seqAxis))
                sequenceX.pop(0)
                sequenceY.pop(0)
                seqAxis.pop(0)
    print len(dataX),len(dataY),len(dataX[0])
    return dataX,dataY,dataAxis
def makeUpSeq3(dataX,dataY,dataAxis,sequence_len,rul=[]):
    dataX = []
    dataY = []
    dataAxis = []
    for i in range(len(data)):
        sequenceX = []
        sequenceY = []
        seqAxis = []
        for j in range(len(data[i])):
            x = data[i][j][1:]
            if len(rul)!=0:
                y = len(data[i])-data[i][j][0]+rul[i]
            else:
                y = len(data[i])-data[i][j][0]
            seqAxis.append(y)  
            y = y if y<130 else 130
            sequenceX.append(x)
            sequenceY.append(y)
            if j>sequence_len-2:
                dataX.append(copy.deepcopy(sequenceX))
                dataY.append(copy.deepcopy(sequenceY))
                dataAxis.append(copy.deepcopy(seqAxis))
                sequenceX.pop(0)
                sequenceY.pop(0)
                seqAxis.pop(0)
    print len(dataX),len(dataY),len(dataX[0])
    return dataX,dataY,dataAxis
def makeUpSeq2(autoencoder,data,sequence_len,rul=[]):
    dataX = []
    dataY = []
    dataAxis = []
    for i in range(len(data)):
        sequenceX = []
        sequenceY = []
        seqAxis = []
        for j in range(len(data[i])):
            x = data[i][j][1:]
            if len(rul)!=0:
                y = len(data[i])-data[i][j][0]+rul[i]
            else:
                y = len(data[i])-data[i][j][0]
            seqAxis.append(y)  
            y = y if y<130 else 130
            
            x = torch.FloatTensor(x)
            encoded, decoded = autoencoder(Variable(x))
            sequenceX.append(encoded.item())
            sequenceY.append(y)
            if j>sequence_len-2:
                
                sequenceX = flowList(sequenceX)
  
                
                dataX.append(copy.deepcopy(sequenceX))
                dataY.append(copy.deepcopy(sequenceY))
                dataAxis.append(copy.deepcopy(seqAxis))
                sequenceX.pop(0)
                sequenceY.pop(0)
                seqAxis.pop(0)
    print len(dataX),len(dataY),len(dataX[0])
    return dataX,dataY,dataAxis
def flowList(Y):
    Y2=[]    
    lenY = len(Y)
    for k in range(lenY):
        res=0
        num=0
        for i in range(k-7,k+7):
            if i<0 or i>=lenY:
                continue
            res = res+Y[i]
            num=num+1
        Y2.append(res/num)    
    return Y2
    
def readrul(filename):
    testrul=[]
    with open(filename) as inp:
        for line in inp.readlines():
            testrul.append(int(line))
    return testrul

def calMSE(lossAll):
    res = 0
    num = 0
    for x in lossAll:
        res = res+pow(x,2)
    for x in res:
        num = num+x
    return pow(num/(len(lossAll)*len(res)),0.5)
    
    
    
def makeUpEncoder(data,sequence_len,rul=[]):
    dataX = []
    dataY = []
    dataAxis = []
    for i in range(len(data)):
        sequenceX = []
        sequenceY = []
        seqAxis = []
        length = len(data[i])
        for j in range(length):
            if j!=0 and j%sequence_len==0:
                dataX.append(copy.deepcopy(sequenceX))
                dataY.append(copy.deepcopy(sequenceY))
                dataAxis.append(copy.deepcopy(seqAxis))
                sequenceX = []
                sequenceY = []
                seqAxis = []
            j = length-1-j
            x = data[i][j][1:]
            if len(rul)!=0:
                y = len(data[i])-data[i][j][0]+rul[i]
            else:
                y = len(data[i])-data[i][j][0]
            seqAxis.append(y)  
            y = y if y<130 else 130
            sequenceX.append(x)
            sequenceY.append(y)
    print len(dataX),len(dataY),len(dataX[0])
    return dataX,dataY,dataAxis
    
def encoderReturn(autoencoder,dataX,dataY,dataAxis):
    X=[]
    Y=[]
    Z=[]
    for (x,y,axis) in zip(dataX,dataY,dataAxis):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        encoded, decoded = autoencoder(Variable(x))
        #Y1= flowList(encoded.detach().numpy())
        X.append(encoded.detach().numpy())
        Y.append(y.detach().numpy())
        Z.append(axis)
    
    dataX = []
    dataY = []
    dataZ=[]
    seqX=[]
    seqY=[]
    seqA=[]
    for i in range(len(X)):
        if i!=0 and Z[i][0] == 0:
            dataX.append(seqX)
            dataY.append(seqY)
            dataZ.append(seqA)
            seqX=[]
            seqY=[]
            seqA=[]
        seqX.extend(X[i])
        seqY.extend(Y[i])
        seqA.extend(Z[i])
    dataX.append(seqX)
    dataY.append(seqY)
    dataZ.append(seqA)  
    print len(dataX),len(dataX[0])  
    return dataX,dataY,dataZ
    
    
