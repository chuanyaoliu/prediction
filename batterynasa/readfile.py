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

#readXLSX()
from scipy.io import loadmat
def readmat():
    cap = []
    X = []
    T1= [0 for i in range(170)]
    T2 =[]
    m = loadmat('./nasa/B0005.mat')
    i = 1
    last = [0,0,0,0,0,0]
    for cricle in m['B0005'][0][0][0][0]:    
        if cricle[0]=='impedance':
            continue  
        gap = np.reshape(cricle[2]-last,(-1))
        if gap[5]<0:
            gap[4]-=1
            gap[5]+=60
        if gap[4]<0:
            gap[3]-=1
            gap[4]+=60
        if gap[3]<0:
            gap[2]-=1
            gap[3]+=24
        if gap[2]<0:
            gap[1]-=1
            gap[2]+=30
        if gap[2]!=0 or gap[1]!=0 or gap[0]!=0 or gap[3]>4:
            res = ((gap[2]*24+gap[3])*60+gap[4])*60+gap[5]
            T1[i]+=(res-np.reshape(cricle[3][0][0][5],(-1))[-1])/3600
            print i,(res-np.reshape(cricle[3][0][0][5],(-1))[-1])/3600,T1[i]
            
        
                  
        last = cricle[2]

        if cricle[0]=='discharge':
            cap.append( np.reshape(cricle[3][0][0][6],(-1))[0])
            T2.append( np.reshape(cricle[3][0][0][5],(-1))[-1])     
            X.append(i)
            
            i+=1
        elif cricle[0]=='charge':
            T1.append( np.reshape(cricle[3][0][0][5],(-1))[-1])
            
            
    plt.plot(X,cap)
    plt.show()


readmat()

