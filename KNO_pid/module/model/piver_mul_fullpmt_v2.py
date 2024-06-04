
import torch.nn as nn
import numpy as np
import torch
from torch.nn import Linear


class piver_mul_fullpmt_v2(nn.Module):
    def __init__(self,**kwargs):
        super(piver_mul_fullpmt_v2, self).__init__()

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



        self.cross_attention_layer = torch.nn.MultiheadAttention(self.hidden_dim, self.n_heads, batch_first=True)

        self.self_attention_layer = torch.nn.MultiheadAttention(self.hidden_dim, self.n_heads, batch_first=True)
 
        # self.cross_attention_layer = perceiver(self.fea, self.num_latents, self.query_dim, self.hidden_dim, self.n_heads, self.dropout_ratio,self.device)
        
        # self.encoderlayer = torch.nn.TransformerEncoderLayer(d_model=self.query_dim, nhead = self.n_heads, dim_feedforward = self.pf_dim, dropout = self.dropout_ratio, activation = "relu",batch_first=True,norm_first=True)
        
        # self.encoder = torch.nn.TransformerEncoder(self.encoderlayer, num_layers=self.n_layers)
        


        self.mlp_cr = torch.nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim, bias=True),

        )

        # self.mlp_sa = torch.nn.Sequential(
        #     nn.LayerNorm(10*self.fea),
        #     nn.Linear(10*self.fea, 10 * self.fea, bias=True),
        #     nn.GELU(),
        #     nn.Linear(10 * self.fea, 10*self.fea, bias=True),

        # )
        self.embedding = nn.Linear(self.fea, self.hidden_dim)



        self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.num_latents*self.hidden_dim, 256),
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
        self.layer_norm_cross = nn.LayerNorm(self.hidden_dim)
        self.layer_norm_cross_latents = nn.LayerNorm(self.hidden_dim)
        self.layer_norm_self = nn.LayerNorm(self.hidden_dim)
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.hidden_dim)).to(self.device)



    def forward(self, data,pos):
        
        x, pos = data, pos


        
        fea = torch.cat([x,pos],dim=2)

        batch_size = fea.shape[0]
        latents = self.latents.repeat(batch_size,1,1)

        fea = self.embedding(fea)
        latents = self.layer_norm_cross_latents(latents)
        fea = self.layer_norm_cross(fea)
        out, _ = self.cross_attention_layer(latents, fea, fea)
        out  = self.mlp_cr(out)

        for i in range(self.n_layers):

            out = self.layer_norm_self(out)

            out, _ = self.self_attention_layer(out, out, out)

            out = self.mlp_cr(out)



        out = torch.reshape(out, (-1,self.num_latents*self.hidden_dim))
        out = self.mlp(out)

            
        return out
