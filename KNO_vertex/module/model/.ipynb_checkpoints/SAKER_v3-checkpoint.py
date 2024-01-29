import torch.nn as nn
import numpy as np
import torch
from torch.nn import Linear



class SAKER_v3(nn.Module):
    def __init__(self,**kwargs):
        super(SAKER_v3, self).__init__()

        self.fea = kwargs['fea']
        self.cla = kwargs['cla']
        self.hidden_dim = kwargs['hidden']
        self.n_layers = kwargs['depths']
        self.n_heads = kwargs['heads']
        self.pf_dim = kwargs['posfeed']
        self.dropout_ratio = kwargs['dropout']
        self.device = kwargs['device']
        self.batch = kwargs['batch']
        self.pmts = kwargs['pmts']
        
        self.dim_model = self.hidden_dim*self.n_heads
        
        self.embedding_1 = nn.Linear(self.fea, self.pf_dim)
        self.embedding_2 = nn.Linear(self.pf_dim, self.dim_model)
        
        self.multihead_attn =  nn.MultiheadAttention(self.dim_model, self.n_heads, batch_first=True)
        
        self.encoderlayer = torch.nn.TransformerEncoderLayer(d_model=self.dim_model, nhead = self.n_heads, dim_feedforward = self.pf_dim, dropout = self.dropout_ratio, activation = "relu",batch_first=True,norm_first=True)
        
        self.encoder = torch.nn.TransformerEncoder(self.encoderlayer, num_layers=self.n_layers)


        self.embedding_3 = nn.Linear(self.dim_model*self.pmts, self.pf_dim)
        self.embedding_4 = nn.Linear(self.pf_dim, self.cla)

        self.linear_top = nn.Sequential(
            nn.Linear(self.dim_model*self.pmts, self.pf_dim),
            nn.LeakyReLU(),
            nn.Linear(self.pf_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.cla),
        )
            


        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.dropout_ratio)        
    def forward(self, data):
        
        
        x, pos, batch = data.x, data.pos, data.batch
        
        x = torch.reshape(x,(-1,self.pmts,2))
        pos = torch.reshape(pos,(-1,self.pmts,3))
        src = torch.cat([x,pos],dim=2)
        
        src = self.dropout(self.relu(self.embedding_1(src)))
        
        src = self.relu(self.embedding_2(src))

        out = self.encoder(src)
        
        out = torch.reshape(out,(-1,self.dim_model*self.pmts))
        
        out = self.linear_top(out)
        
        # out = self.dropout(self.relu(self.embedding_3(out)))
        # out = self.relu(self.embedding_4(out))



            
        return out
