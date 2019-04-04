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

import xlrd
import matplotlib.pyplot as plt
def readTXT():
    path = "./CS2_21/" 
    files= os.listdir(path)
    files.sort()
    s = []
    i = 0
    for file in files:
        #if i==20:
         #   break
        if not os.path.isdir(file): 
            f = open(path+"/"+file);
            iter_f = iter(f); 
            str = ""
            onefile= []
            print file,len(s),i
            for line in iter_f: 
                line = line.split()
                if line[0]=='Time':
                    continue
                line = map(eval,line)
                if line[6]==7:
                    s.append(line) 
                elif line[6]==31:
                    s.append(line)
                    i+=1

    print len(s)

    #for jj in range(29):
    X=[]
    Y=[]
    circle=0
    head = s[0][11]
    for i in range(len(s)):
        if(s[i][6])==31: 
            circle+=1
            #print circle,s[i][13]
            X.append(circle)
            #Y.append(s[i][14])
            Y.append((s[i-1][11]-head)*0.55/3600)
            if i<len(s)-1:
                head=s[i+1][11]

    #Y.sort()    
    #plt.scatter(X, Y, s=15)
    plt.plot(X, Y)
    #plt.scatter(X, Z, s=15)
    #plt.xlim(-10, 300)
    #plt.ylim(0.8, 1.1)
    plt.show()
#readTXT()

def readXLSX():
    path = "./CS2_34/" 
    files= os.listdir(path)
    files.sort()
    data = []
    
    x = []
    count = 0
    for file in files:
        last = [0 for i in range(15)]
        if not os.path.isdir(file): 
            print path+file
            workbook = xlrd.open_workbook(path+file)
            print(workbook.sheet_names())
            if len(workbook.sheet_names())<3:
                continue               
            booksheet = workbook.sheet_by_index(2)    
            
            for i in range(booksheet.nrows):
                if i==0:
                    continue
                line = booksheet.row_values(i)
                temp = line[:]
                for j in (1,5,6,7,8):
                    line[j] = line[j]-last[j]
                line[-2] = line[-2]*0.55/3600
                line[-3] = line[-3]*0.55/3600
                data.append(line)
                last = temp
                count+=1
                x.append(count)

    data = np.asarray(data)

    for j in range(len(data)):
       # plt.scatter(x, data[:,6], s=15)
        plt.plot(x, data[:,j])
        #plt.scatter(X, Z, s=15)
        #plt.xlim(-10, 300)
        #plt.ylim(0.8, 1.15)
        plt.show()      

readXLSX()

