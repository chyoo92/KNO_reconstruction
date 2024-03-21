import torch
import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, input_dim, cross_head, num_latents, query_dim, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()

        assert hidden_dim % n_heads == 0
                
        self.hidden_dim = hidden_dim
        
        self.n_heads = n_heads
        self.cross_head = cross_head
        self.mid_dim = hidden_dim//2
        
        self.query_dim = query_dim
        # before scaled dot-product attention FC
        self.fc_q = nn.Linear(input_dim, query_dim, bias = False)
        self.fc_k = nn.Linear(input_dim, query_dim, bias = False)
        self.fc_v = nn.Linear(input_dim, self.mid_dim, bias = False)



        # each head imbedding dimension
        # self.head_dim = hidden_dim // n_heads
        self.head_dim = query_dim // cross_head

        self.fc_o = nn.Linear(self.mid_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key_value, mask):

        batch_size = key_value.shape[0]
        

        Q = self.fc_q(query)
        K = self.fc_k(key_value)
        V = self.fc_v(key_value)
        
        

        # Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        Q = Q.view(batch_size, -1, self.cross_head, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.cross_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, V.shape[1], self.cross_head, -1).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        mask = mask.unsqueeze(1)
        mask = mask.expand(-1, energy.shape[2], -1)
        mask = mask.unsqueeze(1)
        mask = mask.expand(-1, self.cross_head, -1, -1)
        mask = mask.reshape(batch_size, energy.shape[1], energy.shape[2], energy.shape[3])

        energy = energy.masked_fill(mask==0, float('-inf'))




        attention = torch.softmax(energy, dim=-1)


        x = torch.matmul(self.dropout(attention), V)


        x = x.permute(0, 2, 1, 3).contiguous()

        # print(x.shape,'xxxxxxxxxxxxxxxxx')
        x = x.view(batch_size, -1, self.mid_dim)
        # print(x.shape,'22222222222222222')
        x = self.fc_o(x)


        return x, attention,energy



class FeedforwardLayer(nn.Module):
    def __init__(self, query_dim, dropout_ratio):
        super().__init__()
        
        self.fc_1 = nn.Linear(query_dim, query_dim*2)
        self.fc_2 = nn.Linear(query_dim*2, query_dim)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)

        return x


class perceiverLayer(nn.Module):
    def __init__(self, input_fea, cross_head, num_latents, query_dim, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()
        
        self.self_attention = MultiHeadAttentionLayer(input_fea, cross_head, num_latents, query_dim, hidden_dim, n_heads, dropout_ratio, device)
        
        self.layer_norm_input = nn.LayerNorm(input_fea)
        self.layer_norm_latent = nn.LayerNorm(input_fea)
        
        self.feedforward = FeedforwardLayer(query_dim, dropout_ratio)
        # self.layer_norm = nn.LayerNorm(hidden_dim)
        
        
    def forward(self, latents, src,mask):


        _latents = self.layer_norm_latent(latents)
        _src = self.layer_norm_input(src)
        # _src = self.layer_norm_latent(src)

        _src, _,_ = self.self_attention(_latents, _src,mask)

        # src = self.layer_norm_latent(_latents + _src)
        # src = self.feedforward(src) + src
        src = _src    
        return src



class perceiver(nn.Module):
    def __init__(self, input_fea, cross_head, num_latents, query_dim, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()
        self.num_latents = num_latents
        self.query_dim = query_dim
        self.embedding = nn.Linear(query_dim, hidden_dim)
        self.layers = perceiverLayer(input_fea, cross_head, num_latents, query_dim, hidden_dim, n_heads, dropout_ratio, device)

        self.dropout = nn.Dropout(dropout_ratio)
        self.latents = nn.Parameter(torch.randn(self.num_latents, input_fea)).to(device)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)
        self.device = device
    def forward(self, src,mask):

        
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        
        latents = self.latents.repeat(batch_size,1,1)

        # src = self.embedding(src)
        out = self.layers(latents,src,mask)
        # out = self.embedding(out)

        
        return out