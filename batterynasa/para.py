import math
import torch
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
LR_G = 0.000005      # learning rate for generator
LR_D = 0.001       # learning rate for discriminator
N_IDEAS = 5         # think of this as number of ideas for generating an art work(Generator)
ART_COMPONENTS = 70 # it could be total point G can drew in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])
cricle = 60

def artist_works():    # painting from the famous artist (real target)
    #a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    r = 0.02 * np.random.randn(1, ART_COMPONENTS)
    paintings = np.sin(PAINT_POINTS * np.pi) + r
    paintings = torch.from_numpy(paintings).float()
    return paintings
    
path = './nasa/B0006.mat'
cap,T1,X = readmat(path)
print(len(X),len(cap),len(T1))
maxcap = max(cap)
cap = [float(x)/maxcap for x in cap]

maxT1 = max(T1)
T1 = [float(x)/maxT1 for x in T1]

sequence_len = 10
batch_size = 1
data = makeUpNasa(cap,T1,sequence_len)
print(len(data),len(data[0]))

#batchcap = torch.FloatTensor([[cap[1:71],T1[1:71]]for i in range(BATCH_SIZE)])
#batchinput = torch.FloatTensor([[cap[0:70],T1[0:70]] for i in range(BATCH_SIZE)])

G = nn.Sequential(                  # Generator
    nn.Linear(cricle, 512),        # random ideas (could from normal distribution)
    nn.ReLU(),
    nn.Linear(512, 64),        # random ideas (could from normal distribution)
    nn.ReLU(),
    nn.Linear(64, 3), # making a painting from these random ideas

)
predict = nn.Linear(2,18)   
D = nn.Sequential(                  # Discriminator
    nn.Linear(36, 128), 
    nn.ReLU(),
    nn.Linear(128, 24),
    nn.ReLU(),
    nn.Linear(24, 1),
    #nn.Sigmoid(),                   
)

#net = Net(12, 50, 30,1)
#net = torch.load('./model/model_epoch3000.pkl')
#G = torch.load('./G.pkl')
#opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
#opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
opt_D = torch.optim.SGD(D.parameters(), lr=LR_D)
opt_G = torch.optim.SGD(G.parameters(), lr=LR_G)
loss_func = torch.nn.MSELoss() 
plt.ion()    # something about continuous plotting

D_loss_history = []
G_loss_history = []
for step in range(10000):
    last = torch.FloatTensor(cap[:cricle])
    T =  torch.FloatTensor(T1[:cricle])
    
    fake = G(last)

    for i in range(cricle-1):
        #pre = fake[0]*torch.exp(fake[1]*i)+fake[2]*torch.exp(fake[3]*i)
        if T1[i]!=0:
            pre = fake[0]*last[i]+fake[1]*torch.exp(-fake[2]/T[i])
        else:
            pre = fake[0]*last[i]
        loss= loss_func(pre,last[i+1])
        
        
        opt_G.zero_grad()
        loss.backward(retain_graph=True)    # reusing computational graph
        opt_G.step()

    if step % 50 == 0:  # plotting
        print( step,loss,fake)
        X = []
        Y = []
        Z = cap[:cricle]
        for i in range(cricle):
            #pre = fake[0]*torch.exp(fake[1]*i)+fake[2]*torch.exp(fake[3]*i)
            pre = fake[0]*cap[i]+fake[1]*math.exp(-fake[2]/T1[i])
            Y.append(pre)
            X.append(i)
            i+=1
        plt.cla()
        plt.plot(X,Z, c='#4AD631', lw=3, label='Generated painting',)
        plt.plot(X,Y , c='#74BCFF', lw=3, label='upper bound')
        #plt.text(-1,0.96, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
        #plt.text(-1, 0.84, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        #plt.ylim((0.9, 1.1));
        plt.legend(loc='upper right', fontsize=10);
        plt.draw();plt.pause(0.01)
        
plt.ioff()
plt.show()
print('model has been saved')
