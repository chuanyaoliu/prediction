import torch
import math
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
from util import  *
from model2 import Net 
# torch.manual_seed(1)       # reproducible
# np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 64
LR_G = 0.10       # learning rate for generator
LR_D = 0.001       # learning rate for discriminator
N_IDEAS = 5         # think of this as number of ideas for generating an art work(Generator)
ART_COMPONENTS = 70 # it could be total point G can drew in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])


def artist_works():    # painting from the famous artist (real target)
    #a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    r = 0.02 * np.random.randn(1, ART_COMPONENTS)
    paintings = np.sin(PAINT_POINTS * np.pi) + r
    paintings = torch.from_numpy(paintings).float()
    return paintings
    
path = './nasa/B0005.mat'
cap,T1,X = readmat(path)
print(len(X),len(cap),len(T1))
maxcap = max(cap)
cap = [float(x)/maxcap for x in cap]

maxT1 = max(T1)
T1 = [float(x)/maxT1 for x in T1]

sequence_len = 9
batch_size = 1
data = makeUpNasa(cap,T1,sequence_len)
print(len(data),len(data[0]))

#batchcap = torch.FloatTensor([[cap[1:71],T1[1:71]]for i in range(BATCH_SIZE)])
#batchinput = torch.FloatTensor([[cap[0:70],T1[0:70]] for i in range(BATCH_SIZE)])

G = nn.Sequential(                  # Generator
    nn.Linear(18, 128),        # random ideas (could from normal distribution)
    nn.ReLU(),
    nn.Linear(128, 24),        # random ideas (could from normal distribution)
    nn.ReLU(),
    nn.Linear(24, 1), # making a painting from these random ideas
    nn.Sigmoid(),  
)
'''
predict = nn.Linear(2,18)   
D = nn.Sequential(                  # Discriminator
    nn.Linear(36, 128), 
    nn.ReLU(),
    nn.Linear(128, 24),
    nn.ReLU(),
    nn.Linear(24, 1),
    #nn.Sigmoid(),                   
)
'''
net = Net(12, 50, 30,1)
#net = torch.load('./model/model_epoch3000.pkl')
#G = torch.load('./model/ann2100.pkl')
D = torch.load('./model/ganAD9950.pkl')
predict = torch.load('./model/ganApre9950.pkl')


#opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
#opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
opt_D = torch.optim.SGD(D.parameters(), lr=LR_D)
opt_G = torch.optim.SGD(G.parameters(), lr=LR_G)
loss_func = torch.nn.MSELoss() 
#plt.ion()    # something about continuous plotting

D_loss_history = []
G_loss_history = []
cricle = 70

if cricle ==60:
    a = 0.0417
    b = -0.0083
    c = 1.085
    d = -0.0034
if cricle ==70:
    a = 0.0417
    b = -0.043
    c = 1.0690
    d = -0.0030
def test():
    last = torch.FloatTensor(data[0])
    X = []
    Y = []
    Y2 = []
    Y3=[]
    Z = []
    i = 1
    for line in data[1:]:
        #G_ideas = torch.cat((last,torch.randn(2)))
        #G_ideas = torch.cat((last,torch.FloatTensor([0,0])))
        pre2 = a*math.exp(b*(i+9))+c*math.exp(d*(i+9))
        fake = G(last)
        Y.append(fake.detach().numpy().tolist()[0])
        Y2.append(pre2)
        Z.append(line[-2])
        X.append(i)
        i+=1
        if i<cricle:
            last = torch.FloatTensor(line)
            Y3.append(line[-2])
        else:   
            real = torch.FloatTensor([pre2,line[-1]])
            real_input = torch.cat((last,predict(real)))
            output2 = D(real_input)
            
            real = torch.FloatTensor([fake,line[-1]])
            real = predict(real)
            real_input = torch.cat((last,real))
            output1 = D(real_input)
            
            w1,w2 = softmax(output1.detach().numpy()[0],output2.detach().numpy()[0])
            fusion = fake if abs(output1)<abs(output2) else pre2
            print output1,output2
            Y3.append(fusion)
            
            last = last.detach().numpy().tolist()
            last.pop(0)
            last.pop(0)
            last.append(pre2)
            last.append(line[-1])
            last = torch.FloatTensor(last)

    plt.plot(X,Z, c='#4AD631', lw=3, label='Generated painting',)
    plt.plot(X,Y , c='#74BCFF', lw=3, label='upper bound')
    plt.plot(X,Y2)
    plt.plot(X,Y3)
    plt.show()  
test()      
for step in range(10000):
    last = torch.FloatTensor(data[0])
    X = []
    Y = []
    Z = []
    i = 1
    res=0
    if i%300==0:
        LR_G/=2
        #LR_D/=2
    for line in data[1:]:
        #G_ideas = torch.cat((last,torch.randn(2)))
        #G_ideas = torch.cat((last,torch.FloatTensor([0,0])))
        fake = G(last)
        loss = loss_func(fake,torch.FloatTensor([line[-2]]))
        res+=loss.detach().numpy()
        Y.append(fake.detach().numpy().tolist()[0])
        Z.append(line[-2])
        X.append(i)
        i+=1
        last = torch.FloatTensor(line)
        opt_G.zero_grad()
        loss.backward()
        opt_G.step()
        
    print step,res
    if step % 100 == 0:  # plotting
        #print( step,D_loss,G_loss,prob_artist0,prob_artist1)
        print( step,loss,Y)
        #plt.cla()
        model_name = "./model/ann"+str(step)+".pkl"
        torch.save(G, model_name)
        model_name = "./model/ganpre"+str(step)+".pkl"
        #torch.save(predict, model_name)
        model_name = "./model/ganD"+str(step)+".pkl"
        #torch.save(D, model_name)
        print model_name,"has been saved"
        plt.plot(X,Z, c='#4AD631', lw=3, label='Generated painting',)
        plt.plot(X,Y , c='#74BCFF', lw=3, label='upper bound')
        #plt.text(-1,0.96, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
        #plt.text(-1, 0.84, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        #plt.ylim((0.9, 1.1));
        plt.legend(loc='upper right', fontsize=10);
        #plt.draw();
        #plt.pause(0.01)
        plt.show()
        test()
        
#plt.ioff()
#plt.show()
print('model has been saved')
