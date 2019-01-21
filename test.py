import torch
import numpy as np
import copy
from torch.autograd import Variable
import torch.nn.functional as F 
import matplotlib.pyplot as plt

sequence_len = 30

class Net(torch.nn.Module):  
    def __init__(self, n_feature, n_hidden,n_hidden2, n_output):
        super(Net, self).__init__()   
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)   
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)  
        self.hidden3 = torch.nn.Linear(n_hidden2,50)  
        self.hidden4 = torch.nn.Linear(50,12)  
        self.predict = torch.nn.Linear(300, n_output)   
        self.lstm = torch.nn.LSTM(input_size=n_feature,hidden_size=n_hidden/2,num_layers=1, bidirectional=True)
        self.rnn = torch.nn.RNN(n_hidden2,n_hidden2,1,True)
        self.batch=1
        self.dropout=torch.nn.Dropout(p=0.5)        
        self.n_hidden=n_hidden
        
    def init_hidden_lstm(self):
        return (torch.randn(2, self.batch, self.n_hidden/2),
                torch.randn(2, self.batch, self.n_hidden/2))
    def forward(self, x):   
        '''
        x = F.relu(self.hidden1(x))      
        x = F.relu(self.hidden2(x)) 
        x = F.relu(self.hidden3(x)) 
        x = F.relu(self.hidden4(x)) 
        '''
        self.hidden = self.init_hidden_lstm()
        x = x.view(sequence_len,self.batch,-1)
        x, self.hidden = self.lstm(x, self.hidden)
        '''
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x)) 
        x = F.relu(self.hidden4(x))
        '''
        #x = self.dropout(x)
        x = self.predict(x)   
        return x.view(-1)
net = torch.load('model/model_epoch19.pkl')
def testMSE():
    testdata = []
    oneData=[]
    run = 1
    with open('CMAPSSData/test_FD001.txt') as inp:
        for line in inp.readlines():
            line = line.split()
            if int(line[0])==92:
                oneData.append(map(eval,line))
            elif int(line[0])==93:
                testdata.append(oneData)
                oneData=[]
                oneData.append(map(eval,line))
                run=int(line[0])
                #if run==3:
                break
        testdata.append(oneData)

    print 'test',len(testdata),len(testdata[0])

    testdata = np.asarray(testdata)

    for x in range(len(testdata)):
        meanNum = np.mean(testdata[x],axis=0)
        stdNum = np.std(testdata[x],axis=0)
        for i in range(len(stdNum)):
            if stdNum[i]==0:
                stdNum[i]=1
        meanNum[1]=0
        stdNum[1]=1
        testdata[x]= (testdata[x]-meanNum)/stdNum
        testdata[x] = testdata[x][:,[1,2,3,6,7,8,11,12,13,15,16,17,18,19,21,24,25]]

    testrul=[]
    with open('CMAPSSData/RUL_FD001.txt') as inp:
        for line in inp.readlines():
            testrul.append(int(line))
    print 'testrul',testrul,len(testdata),len(testdata[0]),len(testdata[1])

    
    dataX = []
    dataY = []
    dataAxis = []
    seqAxis = []
    sequenceX = []
    sequenceY = []
    for i in range(len(testdata)):
        sequenceX = []
        sequenceY = []
        seqAxis = []
        for j in range(len(testdata[i])):
            x = testdata[i][j][1:]
            #y = testrul[0]+len(testdata[i])-testdata[i][j][0]
            y =20+len(testdata[i])-testdata[i][j][0]
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
    print len(dataX),len(dataY),len(dataAxis),len(dataX[0])   
    
    X=[]
    Y=[]
    Z=[]
    lossAll = []
    for (x,y,axis) in zip(dataX,dataY,dataAxis):
        x = torch.FloatTensor(x)
        prediction = net(Variable(x))
        #print prediction
        print axis
        X.append(axis)
        Y.append(y)
        Z.append(prediction.detach().numpy())
        lossAll.append(y-prediction.detach().numpy())

    lossAll = np.asarray(lossAll)
    res = 0
    for x in lossAll:
        res = res+pow(x,2)
    print len(lossAll),res/len(lossAll)
    plt.scatter(X, Y, s=15)
    plt.scatter(X, Z, s=15)
    plt.xlim(-10, 300)
    plt.ylim(-10, 300)
    plt.show()
    
    
testMSE()
