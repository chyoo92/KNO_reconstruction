import torch.nn as nn
import numpy as np
import torch
from torch.nn import Linear


from model.perceiver_rev import *

class piver_rev(nn.Module):
    def __init__(self,**kwargs):
        super(piver_rev, self).__init__()

        self.fea = kwargs['fea']
        self.cla = kwargs['cla']
        self.hidden_dim = kwargs['hidden']
        self.n_layers = kwargs['depths']
        self.n_heads = kwargs['heads']
        self.pf_dim = kwargs['posfeed']
        self.dropout_ratio = kwargs['dropout']
        self.device = kwargs['device']
        self.batch = kwargs['batch']
        self.num_latents = kwargs['num_latents']
        self.query_dim = kwargs['query_dim']

 
        self.cross_attention_layer = perceiver(self.fea, self.num_latents, self.query_dim, self.hidden_dim, self.n_heads, self.dropout_ratio,self.device)
        
        self.encoderlayer = torch.nn.TransformerEncoderLayer(d_model=self.query_dim, nhead = self.n_heads, dim_feedforward = self.pf_dim, dropout = self.dropout_ratio, activation = "relu",batch_first=True,norm_first=True)
        
        self.encoder = torch.nn.TransformerEncoder(self.encoderlayer, num_layers=self.n_layers)
        

        self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.num_latents*self.query_dim, 256),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(32, self.cla),
                
            )

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.dropout_ratio)       
    def forward(self, data,pos,mask):
        
        x, pos, mask = data, pos, mask


        
        fea = torch.cat([x,pos],dim=2)

 
        out = self.cross_attention_layer(fea,mask)

        out = self.encoder(out)

        out = torch.reshape(out, (-1,self.num_latents*self.query_dim))
        out = self.mlp(out)

            
        return out
