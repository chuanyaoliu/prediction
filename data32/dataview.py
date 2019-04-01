#coding:utf8
import codecs
import pdb
import sys
import copy
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F     
from torch.autograd import Variable
from util import * 

sequence_len = 30

def dataview(filename):
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
                 #   break
        data.append(oneData)

    print 'train',len(data),len(data[0])

    data = np.asarray(data)
    print len(data[0][0])
    for jj in range(26):
        X=[]
        Y=[]
        for x in range(len(data)):
            meanNum = np.mean(data[x],axis=0)
            stdNum = np.std(data[x],axis=0)
            '''
            for i in range(len(stdNum)):
                if stdNum[i]==0:
                    stdNum[i]=1
            meanNum[1]=0
            stdNum[1]=1
            data[x]= (data[x]-meanNum)#/stdNum
            data[x] = data[x][:,[21]]    #[1,2,6,7,8,11,12,13,15,16,17,18,19,24,25]] 3,21
            '''
            for i in range(len(data[x])):
                X.append(data[x][i][1])
                Y.append(data[x][i][jj])
            
        plt.scatter(X, Y, s=15)
        #plt.plot(X, Y)
        #plt.xlim(-10, 300)
        #plt.ylim(-10, 300)
        plt.show()


dataview('CMAPSSData/train_FD004.txt')

maxNum,minNum = findMaxMin2('CMAPSSData/train_FD002.txt')
data = readFile('CMAPSSData/train_FD002.txt')
data = selectFeature2(data,maxNum,minNum)

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
    plt.show()

