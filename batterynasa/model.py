import torch 
import torch.nn as nn
import torch.nn.functional as F   
class Net(torch.nn.Module):  
    def __init__(self, n_feature, n_hidden, sequence_len,batch_size):
        super(Net, self).__init__()   
        
        self.decoder = torch.nn.Linear(n_hidden, n_feature)
        self.predict = torch.nn.Linear(n_hidden, 1)   
        self.lstm = torch.nn.LSTM(input_size=n_feature,hidden_size=n_hidden/2,num_layers=1, bidirectional=True)
        self.batch=batch_size
        self.sequence_len = sequence_len
        self.dropout=torch.nn.Dropout(p=0.2)        
        self.n_hidden=n_hidden
        self.relu = nn.ReLU()
    def init_hidden_lstm(self):
        return (torch.randn(2, self.batch, self.n_hidden/2),
                torch.randn(2, self.batch, self.n_hidden/2))
    def forward(self, x,train):   
        self.hidden = self.init_hidden_lstm()
        x = x.view(self.sequence_len,self.batch,-1)
        #if train:
        #    x = self.dropout(x)
        x, self.hidden = self.lstm(x, self.hidden)

        x = F.tanh(self.predict(x))
        return x.view(-1)
