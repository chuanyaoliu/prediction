import torch 
import torch.nn as nn
class Net(torch.nn.Module):  
    def __init__(self, n_feature, n_hidden, sequence_len,batch_size):
        super(Net, self).__init__()   
        self.encoder = nn.Sequential(
            nn.Linear(24, 8),
            nn.Tanh(),
            nn.Linear(8, n_feature),   
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_feature, 8),
            nn.Tanh(),
            nn.Linear(8, 24),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )
        self.hidden_linear = torch.nn.Linear(n_hidden, 50)
        self.predict = torch.nn.Linear(n_hidden, 1)   
        self.lstm = torch.nn.LSTM(input_size=n_feature,hidden_size=n_hidden/2,num_layers=1, bidirectional=True)
        self.batch=batch_size
        self.sequence_len = sequence_len
        self.dropout=torch.nn.Dropout(p=0.5)        
        self.n_hidden=n_hidden
    def init_hidden_lstm(self):
        return (torch.randn(2, self.batch, self.n_hidden/2),
                torch.randn(2, self.batch, self.n_hidden/2))
    def forward(self, x):   
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        x = encoded
        self.hidden = self.init_hidden_lstm()
        x = x.view(self.sequence_len,self.batch,-1)
        #x = self.dropout(x)
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.predict(x)   
        return encoded,decoded,x.view(-1)#torch.transpose(x,0,1).view(self.batch,-1)#
