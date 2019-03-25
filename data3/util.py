import copy
import numpy as np
import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

def findMaxMin(filename):
    data = []
    oneData=[]
    maxNum = [0 for i in range(26)]
    #minNum = [100, 362, 0.0087, 0.0006, 100.0, 518.67, 644.53, 1616.91, 1441.49, 14.62, 21.61, 556.06, 2388.56, 9244.59, 1.3, 48.53, 523.38, 2388.56, 8293.72, 8.5848, 0.03, 400, 2388, 100.0, 39.43, 23.6184]
    #minNum = [100, 1, 34.9983, 0.8400, 100.0, 449.44, 555.32, 1358.61, 1137.23, 5.48, 8.00,194.64, 2222.65, 8341.91, 1.02, 42.02, 183.06, 2387.72, 8048.56, 9.3461, 0.02, 334, 2223, 100.0, 14.73, 8.8071]
    #minNum = [100, 1, 0.0008, 0.0004, 100.0, 518.67, 642.36, 1583.23, 1396.84, 14.62, 21.61, 553.97, 2387.96, 9062.17, 1.30, 47.30, 522.31, 2388.01, 8145.32, 8.4246, 0.03, 391, 2388, 100.0, 39.11, 23.3537]
    minNum = [1, 1, 42.0049, 0.8400, 100.0, 445.00, 549.68, 1343.43, 1112.93, 3.91, 5.70, 137.36, 2211.86, 8311.32, 1.01, 41.69, 129.78, 2387.99, 8074.83, 9.3335, 0.02, 330, 2212 ,100.00, 10.62, 6.3670]
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
def findMeanVar(filename):
    data = []
    
    with open(filename) as inp:
        for line in inp.readlines():
            line = line.split()
            line = map(eval,line)
            data.append(line)
    meanNum = np.mean(data,axis=0)
    stdNum = np.std(data,axis=0)
    for i in range(len(stdNum)):
        if stdNum[i]==0:
            stdNum[i]=1
    meanNum[1]=0
    stdNum[1]=1
    return meanNum,stdNum

    
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
                #    break
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
        #data[x] = data[x][:,[1,2,3,6,7,8,11,12,13,15,16,17,18,19,21,24,25]]
        data[x] = data[x][:,1:]

    return data
def selectFeature2(data,meanNum,stdNum):
    data = np.asarray(data)
    meanNum = np.asarray(meanNum)
    stdNum= np.asarray(stdNum)
    
    for x in range(len(data)):
        data[x]= (data[x]-meanNum)/stdNum
        data[x] = data[x][:,[1,2,3,6,7,8,11,12,13,15,16,17,18,19,21,24,25]]

    return data

def makeUpSeq(data,sequence_len,rul=[]):
    dataX = []
    dataY = []
    dataAxis = []
    temp = 130#float(13)/25
    for i in range(len(data)):
        sequenceX = []
        sequenceY = []
        seqAxis = []
        for j in range(len(data[i])):
            x = data[i][j][1:]
            if len(rul)!=0:
                y = float(len(data[i])-data[i][j][0]+rul[i])
            else:
                y = float(len(data[i])-data[i][j][0])
            seqAxis.append(y)  
            
            y = y if y<temp else temp
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
def makeUpSeqTest(data,sequence_len,rul=[]):
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
                y = float(len(data[i])-data[i][j][0]+rul[i])
            else:
                y = float(len(data[i])-data[i][j][0])
            seqAxis.append(y)  
            y = y if y<130 else 130
            sequenceX.append(x)
            sequenceY.append(y)
            if len(sequenceX)==30:
                dataX.append(copy.deepcopy(sequenceX))
                dataY.append(copy.deepcopy(sequenceY))
                dataAxis.append(copy.deepcopy(sequenceX))
                sequenceX = []
                sequenceY = []
                seqAxis = []
    print len(dataX),len(dataY),len(dataX[0])
    return dataX,dataY,dataAxis
def makeUpSeqTest2(data,sequence_len,rul=[]):
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
            
        sequenceX.reverse()
        sequenceY.reverse()
        tx=[]
        ty=[]
        for i in range(len(sequenceX)):
            tx.append(copy.deepcopy(sequenceX[i]))
            ty.append(copy.deepcopy(sequenceY[i]))
            if len(tx)==30:
                tx.reverse()
                ty.reverse()
                dataX.append(copy.deepcopy(tx))
                dataY.append(copy.deepcopy(ty))
                dataAxis.append(copy.deepcopy(tx))
                tx=[]
                ty=[]
                
    print len(dataX),len(dataY),len(dataX[0])
    return dataX,dataY,dataAxis
def makeUpSeqTest3(data,sequence_len,rul=[]):
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
            if len(sequenceX)>30:                
                sequenceX.pop(0)
                sequenceY.pop(0)
                seqAxis.pop(0)
        while len(sequenceX)<30:
            sequenceX.insert(0,sequenceX[0])
            sequenceY.insert(0,sequenceY[0])
            seqAxis.insert(0,seqAxis[0])
        if len(sequenceX)==30:
            dataX.append(copy.deepcopy(sequenceX))
            dataY.append(copy.deepcopy(sequenceY))
            dataAxis.append(copy.deepcopy(seqAxis))
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
    
    
