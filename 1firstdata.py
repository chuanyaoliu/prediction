#coding:utf8
import codecs
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F     # 激励函数都在这
from torch.autograd import Variable

data = []
run = 0
with open('CMAPSSData/train_FD001.txt') as inp:
    for line in inp.readlines():
        line = line.split()
        if int(line[1])>run:
            data.append(map(eval,line))
            run = int(line[1])
        else:
            break

print (data[1])


class Net(torch.nn.Module):  
    def __init__(self, n_feature, n_hidden,n_hidden2, n_output):
        super(Net, self).__init__()   
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)  
        self.hidden3 = torch.nn.Linear(n_hidden2,50)  
        self.hidden4 = torch.nn.Linear(50,12)  
        self.predict = torch.nn.Linear(12, n_output)   
        self.lstm = torch.nn.LSTM(input_size=n_hidden2,hidden_size=n_hidden2,num_layers=1, bidirectional=True)
        self.rnn = torch.nn.RNN(n_hidden2,n_hidden2,1,True)
        self.batch=1
        self.n_hidden2=n_hidden2
    def init_hidden_lstm(self):
        return torch.randn(1, self.batch, self.n_hidden2)
    def forward(self, x):   
      #  self.hidden = self.init_hidden_lstm()
        x = F.relu(self.hidden(x))      
        x = F.relu(self.hidden2(x)) 
        x = F.relu(self.hidden3(x)) 
        x = F.relu(self.hidden4(x)) 
      #  lstm_out, self.hidden = self.lstm(x, self.hidden)
        x = self.predict(x)          
        return x

net = Net(n_feature=20, n_hidden=300,n_hidden2=150, n_output=1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)

X=[]
Y=[]
Z=[]
minNum =100
epoch = 10
for j in range(epoch):
    for i in range(len(data)):
        x = torch.FloatTensor(data[i][6:])
        y = torch.FloatTensor([192-data[i][1]])
        prediction = net(Variable(x))     # 喂给 net 训练数据 x, 输出预测值
        loss = loss_func(prediction,y)/100    # 计算两者的误差
        print loss,y,prediction
        X.append(i)
        Y.append(y.item())
        Z.append(prediction.item())
        if abs(y-prediction)<minNum:
            minNum = abs(y-prediction)
        optimizer.zero_grad()   # 清空上一步的残余更新参数值
        loss.backward()         # 误差反向传播, 计算参数更新值
        optimizer.step()        # 将参数更新值施加到 net 的 parameters 上

print int(minNum)


plt.scatter(X, Y, s=15)
plt.scatter(X, Z, s=15)
plt.xlim(0, 200)
plt.ylim(0, 200)

plt.show()

