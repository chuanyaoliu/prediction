import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from cnn import Net 
import torch
import torch.nn.functional as F  
from torch.autograd import Variable

dataFile = 'mill.mat'
data = scio.loadmat(dataFile)

allVB = [[] for i in range(16)]
allTime = [[] for i in range(16)]
allDoc = [[] for i in range(16)]
allFeed = [[] for i in range(16)]
allMaterial = [[] for i in range(16)]
allAC = [[] for i in range(16)]
allDC = [[] for i in range(16)]
allDC = [[] for i in range(16)]
allDC = [[] for i in range(16)]
allVibTable = [[] for i in range(16)]
allVibSpindle = [[] for i in range(16)]
allAETable = [[] for i in range(16)]
allAESpindle = [[] for i in range(16)]
inputdata = [[] for i in range(16)]


for i in data['mill'][0]: #(167,13)  1,1,1,1,1,1,1,9000,9000,9000,9000,9000,9000
    #print i[0][0][0],i[1],i[2],i[3],i[4],i[5],i[6]
    #for j in i[9][3000:6000]:
    #    allAC[(i[0][0][0]-1)].append(j[0])
    if i[0][0][0]==1 and (i[1][0][0]==16 or i[1][0][0]==17):
        continue
    if i[0][0][0]==7 and i[1][0][0]==8:
        continue
    if i[0][0][0]==12 and i[1][0][0]==1:
        continue
    if i[2][0][0]>=0.45:
        continue
        
    allVB[(i[0][0][0]-1)].append(i[2][0][0])
    allTime[(i[0][0][0]-1)].append(i[3][0][0])
    allDoc[(i[0][0][0]-1)].append(i[4][0][0])
    allFeed[(i[0][0][0]-1)].append(i[5][0][0])
    allMaterial[(i[0][0][0]-1)].append(i[6][0][0])
    allAC[(i[0][0][0]-1)].append([np.mean(i[7][3000:6000]),np.std(i[7][3000:6000],ddof=1)])
    allDC[(i[0][0][0]-1)].append([np.mean(i[8][3000:6000]),np.std(i[8][3000:6000],ddof=1)])
    allVibTable[(i[0][0][0]-1)].append([np.mean(i[9][3000:6000]),np.std(i[9][3000:6000],ddof=1)])
    allVibSpindle[(i[0][0][0]-1)].append([np.mean(i[10][3000:6000]),np.std(i[10][3000:6000],ddof=1)])
    allAETable[(i[0][0][0]-1)].append([np.mean(i[11][3000:6000]),np.std(i[11][3000:6000],ddof=1)])
    allAESpindle[(i[0][0][0]-1)].append([np.mean(i[12][3000:6000]),np.std(i[12][3000:6000],ddof=1)])
    inputdata[(i[0][0][0]-1)].append([np.mean(i[7][3000:6000]),np.std(i[7][3000:6000],ddof=1),np.mean(i[10][3000:6000]),np.std(i[10][3000:6000],ddof=1),np.mean(i[11][3000:6000]),np.std(i[11][3000:6000],ddof=1),np.mean(i[12][3000:6000]),np.std(i[12][3000:6000],ddof=1)])
    #inputdata[(i[0][0][0]-1)].append([np.std(i[7][3000:6000],ddof=1),np.std(i[11][3000:6000],ddof=1),np.std(i[12][3000:6000],ddof=1)])
    
    

print len(allVB[0]),len(allVB[6])

rul = [41,61.8,75.55,36.78,9.33,0,18.67,9.17,32.67,36,76.80,55,20.91,14.4,11.91,6.68]
x=[]
y=[]
last = 0
for j in (4,6,7,12,13,14,15):
    for i in allTime[j]:
        x.append(round(rul[j]-i))
x.sort()

for i in range(len(x)):
    y.append(i+1)
#plt.scatter(y,x,s=10,label='Actual RUL' )
#plt.xlim(0,40)
#plt.ylim(-5,25)
#plt.show()

#for i in range(16):
#    plt.plot(allTime[i],allVB[i],label='Mill'+str(i+1))

x=[]
y=[]
i=0

for jj in range(len(allVibSpindle[15])):
    y.append(allVibSpindle[15][jj][1])
    #for ii in range(len(allAC[4][jj])):
    i+=1
    x.append(i)
#plt.plot(x,y)
#plt.legend(loc='upper right', fontsize=10);
#plt.show()


batch_size=1
seq_len = 3
net = Net(len(inputdata[0][0]),4, 256, seq_len,batch_size)

#net = torch.load('./model/model7_epoch9900.pkl')
optimizer = torch.optim.SGD(net.parameters(), lr=0.00001)
loss_func = torch.nn.MSELoss() 
def findIndex(doc,feed,material):
    index=0
    if material[0]==1:
        if doc[0]==0.75:
            if feed[0]==0.25:    
                index=0
            else:
                index=1
        else:
            if feed[0]==0.25:    
                index=2
            else:
                index=3
    else:
        if doc[0]==0.75:
            if feed[0]==0.25:    
                index=4
            else:
                index=5
        else:
            if feed[0]==0.25:    
                index=6
            else:
                index=7
    return [index for i in range(len(doc))]


def test(index,net):
    X = []
    Y = []
    Z = []
    i = 0
    res = 0
    
    seqX = []
    seqY = []
    fen = findIndex(allDoc[index],allFeed[index],allMaterial[index])
    for (x,y) in zip(inputdata[index],allTime[index]): 
        seqX.append(x)
        seqY.append(round(rul[index] - y))

    encoder,decoder,prediction = net(Variable(torch.FloatTensor(inputdata[index])),Variable(torch.LongTensor(fen)),False) 

    y = torch.FloatTensor(seqY)
    i=i+1 
    loss = loss_func(prediction,y[2:])

    X=[i for i in range(len(seqY)-2)]
    Y=prediction.detach().numpy().tolist()
    Z=seqY[2:]
    res+=loss.detach().numpy() 

    print 'test',index,pow(res,0.5)#,Y,Z

    #plt.plot(X,Y,label='pre' )
    #plt.plot(X,Z,label='true' )
    #plt.legend(loc='upper right', fontsize=10);
    #plt.show()
    return prediction.detach().numpy().tolist(),y[2:].detach().numpy().tolist(),y[:2].detach().numpy().tolist()
def train(index):
    for ep in range(100000):
        '''
        iron = [0,1,2,3,8,9,10,11]
        steel = [4,6,7,12,13,14,15]
        if index in iron:
            train = iron
        elif index in steel:
            train = steel 
        '''
        for j in range(16):
            if j==index or j==5:
                continue
            Y = []
            Z = []
            i = 0
            res = 0
            
            seqX = []
            seqY = []
            fen = findIndex(allDoc[j],allFeed[j],allMaterial[j])
            
            for (x,y) in zip(inputdata[j],allTime[j]):
                seqX.append(x)
                seqY.append(round(rul[j] - y))
                    
                   
            encoded,decoded,prediction = net(Variable(torch.FloatTensor(inputdata[j])),Variable(torch.LongTensor(fen)),True) 

            y = torch.FloatTensor(seqY)
            i=i+1 
            #print prediction.size(),y.size()
            loss1 = loss_func(prediction,y[2:])
            loss2 = loss_func(decoded,torch.FloatTensor(inputdata[j]))
            loss = loss1+loss2
            optimizer.zero_grad()  
            loss1.backward()        
            optimizer.step()
            Y.append(prediction.detach().numpy())
            Z.append(y.detach().numpy())
            res+=loss.detach().numpy() 

            #if ep%500==0:
            #    print ep,'train',pow(res/i,0.5),loss1,loss2
        if ep!=0 and ep%500==0:    
            #plt.plot(allTime[j],Y,label='pre' )
            #plt.plot(allTime[j],Z,label='true' )
            #plt.legend(loc='upper right', fontsize=10);
            #plt.show()
            model_name = "./model/model"+str(index)+"_epoch"+str(ep)+".pkl"
            torch.save(net, model_name)
            print(ep,model_name,"has been saved"  )
            test(index,net)
#train(13)

testmodel = []
testmodel.append(torch.load('./model/model0_epoch49000.pkl'))
testmodel.append(torch.load('./model/model1_epoch99000.pkl'))
testmodel.append(torch.load('./model/model2_epoch39000.pkl'))
testmodel.append(torch.load('./model/model3_epoch63000.pkl'))
testmodel.append(torch.load('./model/model4_epoch25000.pkl'))
testmodel.append(torch.load('./model/model0_epoch49000.pkl'))
testmodel.append(torch.load('./model/model6_epoch25000.pkl'))
testmodel.append(torch.load('./model/model7_epoch13000.pkl'))
testmodel.append(torch.load('./model/model8_epoch21000.pkl'))
testmodel.append(torch.load('./model/model9_epoch30000.pkl'))
testmodel.append(torch.load('./model/model10_epoch99000.pkl'))
testmodel.append(torch.load('./model/model11_epoch37000.pkl'))
testmodel.append(torch.load('./model/model12_epoch11000.pkl'))
testmodel.append(torch.load('./model/model13_epoch28500.pkl'))
testmodel.append(torch.load('./model/model14_epoch46000.pkl'))
testmodel.append(torch.load('./model/model15_epoch15000.pkl'))
steelprediction = []
steeltrue = []
ironprediction = []
irontrue = []
iron = [0,1,2,3,8,9,10,11]
steel = [4,6,7,12,13,14,15]

for i in range(16):
    if i==5:
        continue
    pre, y,wu = test(i,testmodel[i])
    wu2= []
    for x in wu:
        wu2.append(-10)
    ''''    
    if i in iron:
        ironprediction.extend(pre)
        ironprediction.extend(wu2)
        irontrue.extend(y)
        irontrue.extend(wu)
    elif i in steel:
    '''
    steelprediction.extend(pre)
    #steelprediction.extend(wu2)
    steeltrue.extend(y)
    #steeltrue.extend(wu)
    
    
x1=[]
for i in range(len(ironprediction)):
    x1.append(i+1)
x2=[]
for i in range(len(steelprediction)):
    x2.append(i+1)    
    
    
res=0    
rmse=0
for x,y in zip(ironprediction,irontrue):
    cha = float(x)-y
    rmse+=cha*cha
    res+=abs(cha)/y
#print pow(rmse/len(irontrue),0.5),res/len(irontrue)
res=0  
rmse=0  
i=0
for x,y in zip(steelprediction,steeltrue):
    cha =float(x)-y
    rmse+=cha*cha
    if y==0:
        i+=1
        continue
    res+=abs(cha)/y
print pow(rmse/len(steeltrue),0.5),res/(len(steeltrue)-i)



def sortAll(pre,true):
    length = len(true)
    for i in range(1,length):
        for j in range(0,i):
            if true[i-j]<true[i-j-1]:
                temp = true[i-j]
                true[i-j] = true[i-j-1]
                true[i-j-1] = temp
                temp = pre[i-j]
                pre[i-j] = pre[i-j-1]
                pre[i-j-1] = temp
                
'''
sortAll(ironprediction,irontrue) 
plt.scatter(x1,irontrue,s=10,label='Actual RUL' )
plt.scatter(x1,ironprediction,s=15,marker=',',label='Predicted RUL' )
'''
sortAll(steelprediction,steeltrue)  
plt.scatter(x2,steeltrue,s=10,label='Actual RUL' )
plt.scatter(x2,steelprediction,s=15,marker=',',label='Predicted RUL'  )

plt.legend(loc='upper left', fontsize=10);
plt.tick_params(direction='in')
plt.xlabel('Increasing RUL')
plt.ylabel('RUL')
#plt.xlim(0,100)
#plt.ylim(0,80)
plt.savefig('570.jpg', dpi=800)
plt.show()

