
import torch.nn as nn
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(16, 4),
            nn.Tanh(),
            nn.Linear(4, 1),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 4),
            nn.Tanh(),
            nn.Linear(4, 16),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
#autoencoder = AutoEncoder()

