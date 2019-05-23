import copy
import numpy as np
import torch
import math
from torch.autograd import Variable

import matplotlib.pyplot as plt
def calScore(pre,y):
    d = pre-y
    if d<0:
        res=math.exp(-d/13)
    else:
        res=math.exp(d/10)   
    return res
def findindex(data):
    rou = round(data)
    if rou==0:
        rou=0
    elif rou==10:
        rou=1
    elif rou==20:
        rou=2
    elif rou==25:
        rou=3
    elif rou==35:
        rou=4
    elif rou==42:
        rou=5
    return rou
        
def findMaxMin(filename):
    data = []
    oneData=[]
    maxNum = [0 for i in range(26)]
    #minNum = [100, 362, 0.0087, 0.0006, 100.0, 518.67, 644.53, 1616.91, 1441.49, 14.62, 21.61, 556.06, 2388.56, 9244.59, 1.3, 48.53, 523.38, 2388.56, 8293.72, 8.5848, 0.03, 400, 2388, 100.0, 39.43, 23.6184]
    #minNum = [100, 1, 34.9983, 0.8400, 100.0, 449.44, 555.32, 1358.61, 1137.23, 5.48, 8.00,194.64, 2222.65, 8341.91, 1.02, 42.02, 183.06, 2387.72, 8048.56, 9.3461, 0.02, 334, 2223, 100.0, 14.73, 8.8071]
    minNum = [100, 1, 0.0008, 0.0004, 100.0, 518.67, 642.36, 1583.23, 1396.84, 14.62, 21.61, 553.97, 2387.96, 9062.17, 1.30, 47.30, 522.31, 2388.01, 8145.32, 8.4246, 0.03, 391, 2388, 100.0, 39.11, 23.3537]
    #minNum = [1, 1, 42.0049, 0.8400, 100.0, 445.00, 549.68, 1343.43, 1112.93, 3.91, 5.70, 137.36, 2211.86, 8311.32, 1.01, 41.69, 129.78, 2387.99, 8074.83, 9.3335, 0.02, 330, 2212 ,100.00, 10.62, 6.3670]
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
def findMaxMin2(filename):
    data = []
    oneData=[]
    maxNum = [[0 for i in range(26)],
            [0 for i in range(26)],
            [0 for i in range(26)],
            [0 for i in range(26)],
            [0 for i in range(26)],
            [0 for i in range(26)]]
    minNum = [[10000 for i in range(26)] for j in range(6)]
    '''
    minNum = [[1, 62, 0.0029, 0.0000, 100.0, 518.67, 643.24, 1591.46, 1405.31 ,14.62, 21.61, 553.34 ,2388.14, 9040.41, 1.30, 47.54, 521.83, 2388.21, 8117.75, 8.4424, 0.03, 392, 2388, 100.00 ,38.83, 23.1994  ],
    [1, 35, 10.0066, 0.2510 ,100.0, 489.05, 605.25, 1504.23, 1306.63, 10.52, 15.49, 394.24, 2319.01, 8765.65, 1.26 ,45.48, 371.35, 2388.15, 8114.43, 8.6566, 0.03, 368, 2319 ,100.00, 28.74 ,17.1825  ],
    [1, 8 ,20.0020, 0.7002, 100.0, 491.19, 607.44, 1481.69, 1252.36, 9.35, 13.65, 334.41, 2323.87, 8709.12, 1.08, 44.27, 315.11, 2387.99, 8049.26, 9.2369, 0.02 ,365, 2324, 100.00, 24.33, 14.7989  ],
    [1, 3, 24.9988, 0.6218, 60.0, 462.54, 537.31 ,1256.76, 1047.45, 7.05, 9.02, 175.71 ,1915.11, 8001.42, 0.94, 36.69, 164.22, 2028.03, 7864.87, 10.8941, 0.02, 309, 1915, 84.93, 14.08, 8.6723  ],
    [100, 1, 34.9983, 0.8400, 100.0, 449.44, 555.32, 1358.61, 1137.23, 5.48, 8.00,194.64, 2222.65, 8341.91, 1.02, 42.02, 183.06, 2387.72, 8048.56, 9.3461, 0.02, 334, 2223, 100.0, 14.73, 8.8071],
    [1, 2, 41.9982, 0.8408, 100.0 ,445.00, 549.90, 1353.22, 1125.78, 3.91, 5.71, 138.51, 2211.57, 8303.96, 1.02 ,42.20 ,130.42, 2387.66 ,8072.30, 9.3774, 0.02, 330, 2212, 100.00, 10.41, 6.2665]]
    '''
    with open(filename) as inp:
        for line in inp.readlines():
            line = line.split()
            line = map(eval,line)
            rou = findindex(line[2])
            
            for i in range(26):
                if line[i]>maxNum[rou][i]:
                    maxNum[rou][i]=line[i]
                if line[i]<minNum[rou][i]:
                    minNum[rou][i]=line[i]
            
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
                #if run==2:
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
        #data[x] = data[x][:,1:]

    return data
def selectFeature2(data,maxNum,minNum):
    data = np.asarray(data)
    condition = []
    maxNum = np.asarray(maxNum)
    minNum= np.asarray(minNum)
    for x in range(len(data)):
        cond = []
        for i in range(len(maxNum)):   
            for j in range(len(maxNum[0])):
                if maxNum[i][j]==minNum[i][j]:
                    maxNum[i][j]=maxNum[i][j]+1   
            minNum[i][1]=0
            maxNum[i][1]=1
        for y in range(len(data[x])):
            rou = findindex(data[x][y][2])
            cond.append(rou)
            data[x][y]= (data[x][y]-minNum[rou])/(maxNum[rou]-minNum[rou])
            data[x][y]= [data[x][y][p] for p in[1,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,24,25]]    
        #data[x] = data[x][:,[1,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,24,25]]   
        condition.append(cond[:])     
    return data,condition    

def selectFeature3(data,meanNum,stdNum):
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
def makeUpSeq2(data,sequence_len,rul=[]):
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
        dataX.append(copy.deepcopy(sequenceX))
        dataY.append(copy.deepcopy(sequenceY))
        dataAxis.append(copy.deepcopy(seqAxis))
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
    
    
