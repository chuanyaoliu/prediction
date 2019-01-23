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
import torch.nn.functional as F     # 激励函数都在这
from torch.autograd import Variable


'''
b = np.array([[[2,3,5],
             [2,3,5],
             [2,3,5]],
             [[2,3,5],
             [2,3,5],
             [2,3,5]]])
a = np.array([1,2,3])
print a[[0,2]]
'''
sequence_len = 30


data = []
oneData=[]
run = 1
with open('CMAPSSData/train_FD001.txt') as inp:
    for line in inp.readlines():
        line = line.split()
        if int(line[0])==run:
            oneData.append(map(eval,line))
        else:
            data.append(oneData)
            oneData=[]
            oneData.append(map(eval,line))
            run=int(line[0])
            if run==3:
                break
    data.append(oneData)

print 'train',len(data),len(data[0])

data = np.asarray(data)
X = []
Y=[]
for x in range(len(data)):
    maxNum = np.max(data[x],axis=0)
    minNum = np.min(data[x],axis=0)
    
    for i in range(len(maxNum)):
        if maxNum[i]==minNum[i]:
            maxNum[i]=maxNum[i]+1
    minNum[1]=0
    maxNum[1]=1
    data[x]= (data[x]-minNum)/(maxNum-minNum)
    X.append(data[x][:,1])
    data[x] = data[x][:,[2,3,6,7,8,11,12,13,15,16,17,18,19,21,24,25]]
    
    scatter_matrix = np.dot(data[x].T,data[x])
    eig_val,eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(16)]
    #eig_pairs.sort(reverse=True)
    # 选取前n个特征向量
    feature = np.array([ele[1] for ele in eig_pairs[: 2]])
    # 转化得到降维的数据
    new_data = np.dot(data[x], feature.T)
    print new_data.shape
    Y.append(new_data)
for i in range(len(X)):
    plt.plot(X[i],Y[i])
#plt.scatter(X, Y, s=10)
plt.show()


