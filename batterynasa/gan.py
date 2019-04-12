import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from util import  *
from model2 import Net 
# torch.manual_seed(1)       # reproducible
# np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 64
LR_G = 0.00001       # learning rate for generator
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
    
path = './nasa/B0006.mat'
cap,T1,X = readmat(path)
print len(X),len(cap),len(T1)
maxcap = max(cap)
cap = [float(x)/maxcap for x in cap]
maxT1 = max(T1)
T1 = [float(x)/maxT1 for x in T1]

sequence_len = 10
batch_size = 1
data = makeUpNasa(cap,T1,sequence_len)
print len(data),len(data[0])

#batchcap = torch.FloatTensor([[cap[1:71],T1[1:71]]for i in range(BATCH_SIZE)])
#batchinput = torch.FloatTensor([[cap[0:70],T1[0:70]] for i in range(BATCH_SIZE)])

G = nn.Sequential(                  # Generator
    nn.Linear(2, 50),        # random ideas (could from normal distribution)
    nn.ReLU(),
    
    nn.Linear(50, 1), # making a painting from these random ideas
)

D = nn.Sequential(                  # Discriminator
    nn.Linear(10, 128), # receive art work either from the famous artist or a newbie like G
    nn.ReLU(),
    nn.Linear(128, 1),
    #nn.Sigmoid(),                   # tell the probability that the art work is made by artist
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
    last = torch.FloatTensor(data[0])
    X = []
    Y = []
    Z = []
    i = 1
    for line in data[1:40]:
        G_ideas = torch.randn(10, 2)
        G_paintings = G(G_ideas).view(-1)    

        Y.append(G_paintings.detach().numpy()[-1])
        Z.append(line[-1][0])
        X.append(i)
        i+=1
        
        
        last = torch.FloatTensor(line)
        
        pre = []
        Dinput = []
        for (a,b) in zip(G_paintings,line):
            pre.append(a)
            pre.append(b[1])
            Dinput.append(b[0])
            
           
        
        Dinput = torch.FloatTensor(Dinput)
        prob_artist0 = D(Dinput)         # D try to increase this prob
        prob_artist1 = D(G_paintings)              # D try to reduce this prob

        #D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
        #G_loss = torch.mean(torch.log(1. - prob_artist1))
        #G_loss = loss_func(G_paintings,torch.FloatTensor([x]))
        D_loss = - torch.mean((prob_artist0) - prob_artist1)
        G_loss = torch.mean( - prob_artist1)
        
        #G_loss_history.append(G_loss)
        
        opt_D.zero_grad()
        D_loss.backward(retain_graph=True)    # reusing computational graph
        opt_D.step()
        if i%5==0:
            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()

    if step % 10 == 0:  # plotting
        print step,D_loss,G_loss,prob_artist0,prob_artist1
        plt.cla()
        plt.plot(X,Z, c='#4AD631', lw=3, label='Generated painting',)
        plt.plot(X,Y , c='#74BCFF', lw=3, label='upper bound')
        plt.text(-1, 1, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
        plt.text(-1, 0.95, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        #plt.ylim((0.9, 1.1));
        plt.legend(loc='upper right', fontsize=10);
        plt.draw();plt.pause(0.01)
        
plt.ioff()
plt.show()
torch.save(G, './G.pkl')
torch.save(D, './D.pkl')
print('model has been saved')
