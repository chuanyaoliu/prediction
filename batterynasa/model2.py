import torch 
import torch.nn as nn
import torch.nn.functional as F   
class Net(torch.nn.Module):  
    def __init__(self, n_feature, n_hidden, sequence_len,batch_size):
        super(Net, self).__init__()   
        
        self.linear1 = torch.nn.Linear(2,50)
        self.linear2 = torch.nn.Linear(50, 100)
        self.predict = torch.nn.Linear(100,1)   
        self.lstm = torch.nn.LSTM(input_size=n_feature,hidden_size=n_hidden/2,num_layers=1, bidirectional=True)
        self.batch=batch_size
        self.sequence_len = sequence_len
        self.dropout=torch.nn.Dropout(p=0.2)        
        self.n_hidden=n_hidden
        self.relu = nn.ReLU()
        self.b1 =  nn.Parameter(torch.FloatTensor([0.1641]))
        self.b2 =  nn.Parameter(torch.FloatTensor([1.6023]))
        self.n =nn.Parameter( torch.FloatTensor([0.9952]))
        self.w =nn.Parameter( torch.FloatTensor([0.6023]))
    def init_hidden_lstm(self):
        return (torch.randn(2, self.batch, self.n_hidden/2),
                torch.randn(2, self.batch, self.n_hidden/2))
    def forward(self, c,train):   
        y = self.relu(self.linear1(c))

        y = self.relu(self.linear2(y))

        y = self.relu(self.predict(y))

        #y = self.n*c+self.b1*torch.exp(-self.b2/t)+self.w
        #y = self.n*torch.exp(self.b1*k)+self.b2*torch.exp(self.w*k)
        return y
