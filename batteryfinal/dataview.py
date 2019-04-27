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


sequence_len = 30

def dataview(filename):
    data = []
    oneData=[]
    run = 1
    with open(filename) as inp:
        for line in inp.readlines():
            line = line.split()
            if line[0]=='Time':
                continue
            line = map(eval,line)
            if line[6]==7:
                data.append(line) 
            elif line[6]==31:
                data.append(line)    
    print 'train',len(data),len(data[0])
    #data.pop(0)
    
    data = np.asarray(data)
    print data.shape
    X=[]
    Y=[]
    circle=0
    head = data[0][11]
    for i in range(len(data)):
        if(data[i][6])==31: 
            circle+=1
            #print circle,data[i][13]
            X.append(circle)
            Y.append((data[i-1][11]-head)*0.55/3600)
            if i<len(data)-1:
                head=data[i+1][11]

    #Y.sort()    
    plt.scatter(X, Y, s=10)
    #plt.plot(X, Y)
    #plt.scatter(X, Z, s=15)
    #plt.xlim(-10, 300)
    plt.ylim(0.7, 1.2)
    plt.show()


#dataview('CS2_8/CS2_8_3_5_10.txt')
def viewXLSX():
    path = "./cs33/" 
    file = "CS2_33_1_10_11.xlsx"
    workbook = xlrd.open_workbook(path+file)
    print(workbook.sheet_names())                  
    booksheet = workbook.sheet_by_index(2)    
    for j in range(booksheet.ncols):
        x=[]
        y=[]
        for i in range(booksheet.nrows):
            if i==0:
                continue
            line = booksheet.row_values(i)
            x.append(i)
            y.append(line[j])
        plt.scatter(x, y, s=15)
        #plt.plot(X, Y)
        #plt.scatter(X, Z, s=15)
        #plt.xlim(-10, 300)
        #plt.ylim(0.8, 1.15)
        plt.show()    

viewXLSX()    
