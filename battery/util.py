#coding:utf8
import codecs
import pdb
import sys
import copy
import pandas as pd
import xlrd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F     
from torch.autograd import Variable
import os
from datetime import datetime
import xlrd
from xlrd import xldate_as_tuple
import matplotlib.pyplot as plt

def readXLSX(path):
    files= os.listdir(path)
    files.sort()
    data = []
    
    x = []
    count = 0
    lastdate = '2010-10-18 16:22:33'
    for file in files:
        if not os.path.isdir(file): 
            print path+file
            workbook = xlrd.open_workbook(path+file)
            print(workbook.sheet_names())
            if len(workbook.sheet_names())<3:
                continue                   
            booksheet = workbook.sheet_by_index(2)    
            last = [0 for i in range(15)]
            for i in range(booksheet.nrows):
                if i==0:
                    continue
                line = booksheet.row_values(i)
                date = xldate_as_tuple(line[2],0)
                line[2] = str(datetime(*date))
                temp = line[:]
                for j in (1,5,6,7,8):
                    line[j] = line[j]-last[j]
                
                diff = datetime.strptime(line[2], '%Y-%m-%d %H:%M:%S') -datetime.strptime(lastdate, '%Y-%m-%d %H:%M:%S')
                lastdate=line[2]
                line[2] = diff.days
                data.append(line)
                last = temp
                count+=1
                x.append(count)
    data = np.asarray(data)
    data = data[:,(1,2,3,4,5,6,7,8,9,12,13,14)]
    #data = data[:,5]
    return data     
def writeData(data,path):
    output_data = codecs.open(path,'w','utf-8')
    for i in range(len(data)):
        #output_data.write(str(data[i])+'\n')
        output_data.write(str(data[i].tolist())[1:-1]+'\n')
        
def readData(path):
    input_data = codecs.open(path,'r','utf-8')
    data = []
    maxNum = [-10 for i in range(12)]
    minNum = [100000 for i in range(12)]
    index = 0
    last=0
    for line in input_data.readlines():
        line=line.strip()
        line = line.split(',')
        line = map(eval,line)
        index+=1
        if index in [43,62,132,173,189,221,234,247,354,469,496,502,510,524]:
            last = line[1]
            continue
        if last!=0:
            line[1]=last
            last=0
        data.append(line)
        for i in range(len(line)):
            if line[i]>maxNum[i]:
                maxNum[i]=line[i]
            if line[i]<minNum[i]:
                minNum[i]=line[i]
    return data,maxNum,minNum
    
def makeUpSeq(data,maxNum,minNum,seqLen):
    data = np.asarray(data)
    maxNum = np.asarray(maxNum)
    minNum= np.asarray(minNum)
    dataX = []
    dataY = []
    x = []
    y = []
    for i in range(len(data)):
        for j in range(len(maxNum)):
            if maxNum[j]==minNum[j]:
                maxNum[j]+=1
        data[i]= (data[i]-minNum)/(maxNum-minNum)
        x.append(data[i][:])
        y.append(data[i][5])
        if i>seqLen-2:
            dataX.append(x[:])
            dataY.append(y[:])
            x.pop(0)
            y.pop(0)
    return dataX,dataY        
def makeUpSeq2(data,maxNum,minNum,seqLen):
    data = np.asarray(data)
    dataX = []
    dataY = []
    x = []
    y = []
    last = data[0]
    for i in range(len(data)):
        x.append(last)
        y.append(data[i])
        last = data[i]
        if i>seqLen-2:
            dataX.append(x[:])
            dataY.append(y[:])
            x.pop(0)
            y.pop(0)
    print len(dataX),len(dataY),len(dataX[10])
    return dataX,dataY           
