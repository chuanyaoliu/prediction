import torch 
import torch.nn as nn
import torch.nn.functional as F
class Net(torch.nn.Module):  
    def __init__(self, n_feature,embedding_dim, n_hidden, sequence_len,batch_size):
        super(Net, self).__init__()   
        self.encoder = nn.Sequential(
            nn.Linear(8, 6),
            nn.Tanh(),
            nn.Linear(6,4),   
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 6),
            nn.Tanh(),
            nn.Linear(6, 8),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )
        self.n_feature = n_feature
        self.embedding_dim = embedding_dim
        self.embedding_dim2 = 0
        self.conv_kernel_num = 10
        self.conv1 = nn.Conv2d(1,self.conv_kernel_num, kernel_size=(3,self.n_feature+self.embedding_dim),stride=1,padding=0)#,dilation=(3,1))
        self.condition_embeds = nn.Embedding(8,self.embedding_dim)
        self.hidden_linear = torch.nn.Linear(n_hidden, 50)
        self.predict = torch.nn.Linear(self.conv_kernel_num, 1)   
        self.lstm = torch.nn.LSTM(input_size=self.n_feature+self.embedding_dim,hidden_size=n_hidden/2,num_layers=1, bidirectional=True)
        self.batch=batch_size
        self.sequence_len = sequence_len
        self.dropout=torch.nn.Dropout(p=0.5)        
        self.n_hidden=n_hidden
    def init_hidden_lstm(self):
        return (torch.randn(2, self.batch, self.n_hidden/2),
                torch.randn(2, self.batch, self.n_hidden/2))
    def forward(self, x,condition,train):   
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        #x = encoded
        embeds = self.condition_embeds(condition)
        x = torch.cat((embeds,x),1)
        x=x.view(self.batch,1,-1,self.n_feature+self.embedding_dim)
        #print x.size()
        cnn_out = F.relu(self.conv1(x))
        #print cnn_out.size()
        cnn_out = cnn_out.view(self.conv_kernel_num,-1)
        cnn_out = torch.transpose(cnn_out,0,1)
        x = self.predict(cnn_out)   

        return encoded,decoded,x.view(-1)#torch.transpose(x,0,1).view(self.batch,-1)#
