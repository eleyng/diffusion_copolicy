import pdb
from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
RED_SIZE = 64
SIZE = 64

def sample_discrete(probs):
    m = Categorical(probs)
    action = m.sample()
    return action

class BC(nn.Module):
    def __init__(self, latents, hiddens, actions, device):
        super().__init__()
        self.device = device
        self.latents = latents
        self.dropout = nn.Dropout(0.2)
        self.mlp = nn.Sequential(
                    nn.Linear(latents, hiddens),
                    nn.Dropout(0.2),
                    nn.LeakyReLU(),
                    nn.Linear(hiddens, 2*hiddens),
                    nn.LeakyReLU(),
                    nn.Linear(2*hiddens, 2*hiddens),
                    nn.LeakyReLU(),
                    nn.Linear(2*hiddens, hiddens),
                    nn.LeakyReLU(),
                    nn.Linear(hiddens, actions),
                    nn.LeakyReLU(),
                    )
         
    def forward(self, latents):
        o = self.mlp(latents) 
        return o

class BCRNN(nn.Module):
    def __init__(self, latents, hiddens, actions, n_layers, device, n_bins=None):
        super().__init__()
        self.device = device
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.dropout = nn.Dropout(0.2)
        self.emb = nn.Linear(latents, hiddens)
        #self.gru = nn.GRUCell(hiddens, 2*hiddens, n_layers)
        self.gru = nn.GRU(hiddens, 2*hiddens, n_layers)
        self.mlp = nn.Linear(2*hiddens, actions)
        '''self.mlp = nn.Sequential(
                    nn.Linear(2*hiddens, hiddens),
                    nn.LeakyReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hiddens, hiddens),
                    nn.LeakyReLU(),
                    nn.Linear(hiddens, actions),
                    nn.LeakyReLU(),
                    )'''
        # Hardmaru MDN
        if n_bins is not None:
            self.n_bins = n_bins
        else:
            self.n_bins = 1
        self.z_pi = nn.Linear(2*hiddens,  self.n_bins)
 
    def forward(self, latents, hiddens):
        
        o = self.emb(latents)
        o, hn = self.gru(o, hiddens)
        o = self.z_pi(o)
        o = o.view(-1, self.n_bins) # logits
        log_probs = f.log_softmax(o, dim=-1)
        probs = torch.exp(log_probs)
        #print('output:' , o.shape, '\nprobs:', probs[0, :])
        #pi = self.z_pi(o)
        #pi = pi.view(-1, self.n_bins, self.actions) # logits for each act
        #logpi = f.log_softmax(pi, dim=-1) # softmax (added log for stability) for logits to probs conversion
        #probs = torch.exp(logpi) # probs for sampling
        
        #return pi, probs, hn
        return o, hn, probs

class aeBCRNN(nn.Module):
    def __init__(self, latents, hiddens, actions, n_layers, device):
        super().__init__()
        self.device = device
        self.latents = latents
        self.dropout = nn.Dropout(0.2)
        
        # Autoencoder
        self.img_channels = 3
        # Encoding
        self.conv1 = nn.Conv2d(self.img_channels, 32, 4, stride=2) # (in_ch=3, out_ch, kernel, ...)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc_z = nn.Linear(2*2*256, self.latents)
        
        self.lstm = nn.LSTM(latents, hiddens, n_layers)
        self.mlp = nn.Sequential(
                    nn.Linear(hiddens, hiddens),
                    nn.LeakyReLU(),
                    nn.Linear(hiddens, hiddens),
                    nn.LeakyReLU(),
                    nn.Linear(hiddens, actions),
                    nn.LeakyReLU(),
                    )
 
    def forward(self, x):
        #print(x.size())
        seq_len, bsize, _, _, _ = x.size()

        x = f.upsample(x.contiguous().view(-1, 3, SIZE, SIZE), size=RED_SIZE,
                       mode='bilinear', align_corners=True)
        #print('a', x.size())
        

        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = f.sigmoid(self.conv4(x))
        
        x = x.view(seq_len, bsize, -1)
        #print(x.size())
        z = self.fc_z(x)

        o, _ = self.lstm(z)
        o = self.mlp(o)

        return o


