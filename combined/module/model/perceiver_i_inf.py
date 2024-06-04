
import torch.nn as nn
import numpy as np
import torch
from torch.nn import Linear
from einops.layers.torch import Reduce

class perceiver_i_inf(nn.Module):
    def __init__(self,**kwargs):
        super(perceiver_i_inf, self).__init__()
        
        self.fea = kwargs['fea']
        self.cla = kwargs['cla']
        self.cross_head = kwargs['cross_head']
        self.cross_dim = kwargs['cross_dim']
        self.self_head = kwargs['self_head']
        self.self_dim = kwargs['self_dim']
        self.n_layers = kwargs['n_layers']
        self.num_latents = kwargs['num_latents']
        self.dropout_ratio = kwargs['dropout_ratio']
        self.batch = kwargs['batch']
        self.device = kwargs['device']


        self.cross_attention_layer = torch.nn.MultiheadAttention(self.cross_dim, self.cross_head, dropout = self.dropout_ratio, batch_first=True)
        self.self_attention_layer = torch.nn.MultiheadAttention(self.self_dim, self.self_head, dropout = self.dropout_ratio, batch_first=True)

        self.mlp_cr = torch.nn.Sequential(
            nn.LayerNorm(self.cross_dim),
            nn.Linear(self.cross_dim, 2 * self.cross_dim, bias=True),
            nn.GELU(),
            nn.Linear(2 * self.cross_dim, self.cross_dim, bias=True),
        )


        self.mlp_sl = torch.nn.Sequential(
            nn.LayerNorm(self.self_dim),
            nn.Linear(self.self_dim, 2 * self.self_dim, bias=True),
            nn.GELU(),
            nn.Linear(2 * self.self_dim, self.self_dim, bias=True),
        )

        self.cr_sl = nn.Linear(self.cross_dim, self.self_dim)

        self.embedding = nn.Linear(self.fea, self.cross_dim)

        self.to_logits = torch.nn.Sequential(
                                            Reduce('b n d -> b d', 'mean'),
                                            nn.LayerNorm(self.self_dim),
                                            nn.Linear(self.self_dim,self.cla))


        self.layer_norm_cross = nn.LayerNorm(self.cross_dim)
        self.layer_norm_cross_latents = nn.LayerNorm(self.cross_dim)

        self.layer_norm_self = nn.LayerNorm(self.self_dim)

        self.dropout = nn.Dropout(self.dropout_ratio)   

        self.latents = nn.Parameter(torch.randn(self.num_latents, self.cross_dim)).to(self.device)



    def forward(self, data,pos,padding_index = None):
        if padding_index is not None:
            
            key_padding_mask = padding_index
        else:
            key_padding_mask = None
        x, pos = data, pos

        
        fea = torch.cat([x,pos],dim=2)

        batch_size = fea.shape[0]

        latents = self.latents.repeat(batch_size,1,1)



        fea = self.embedding(fea)



        latents = self.layer_norm_cross_latents(latents)
        out = self.layer_norm_cross(fea)
        
        out, _ = self.cross_attention_layer(latents, out, out, key_padding_mask=key_padding_mask)
        out = self.layer_norm_cross(self.dropout(out)  + latents)
        out  = self.dropout(self.mlp_cr(out)) + out
        out = self.layer_norm_cross(out)

        out = self.cr_sl(out)
        
        for i in range(self.n_layers):

            out = self.layer_norm_self(out)
            _out, _ = self.self_attention_layer(out, out, out)
            out = self.layer_norm_self(self.dropout(_out) + out)
            out = self.dropout(self.mlp_sl(out))+out
            out = self.layer_norm_self(out)

        
        out = self.to_logits(out)

            
        return out
