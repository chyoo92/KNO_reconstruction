import torch
import torch.nn as nn
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, input_dim, num_latents, query_dim, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()

        assert hidden_dim % n_heads == 0
                
        self.hidden_dim = hidden_dim
        
        self.n_heads = n_heads

        
        
        # before scaled dot-product attention FC
        self.fc_q = nn.Linear(query_dim, hidden_dim, bias = False)
        self.fc_kv = nn.Linear(input_dim, hidden_dim * 2, bias = False)
        # self.fc_kv = nn.Linear(query_dim, hidden_dim * 2, bias = False)
        


        # each head imbedding dimension
        self.head_dim = hidden_dim // n_heads 

        self.fc_o = nn.Linear(hidden_dim, query_dim)

        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key_value):

        batch_size = key_value.shape[0]
        
        # print(key_value.shape)
        # print(key_value)
        
        Q = self.fc_q(query)
        K, V = self.fc_kv(key_value).chunk(2, dim = -1)


        

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale


        attention = torch.softmax(energy, dim=-1)


        x = torch.matmul(self.dropout(attention), V)


        x = x.permute(0, 2, 1, 3).contiguous()


        x = x.view(batch_size, -1, self.hidden_dim)

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
    def __init__(self, input_fea, num_latents, query_dim, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()
        
        self.self_attention = MultiHeadAttentionLayer(input_fea, num_latents, query_dim, hidden_dim, n_heads, dropout_ratio, device)
        
        self.layer_norm_input = nn.LayerNorm(input_fea)
        self.layer_norm_latent = nn.LayerNorm(query_dim)
        
        self.feedforward = FeedforwardLayer(query_dim, dropout_ratio)
        # self.layer_norm = nn.LayerNorm(hidden_dim)
        

    def forward(self, latents, src):


        _latents = self.layer_norm_latent(latents)
        _src = self.layer_norm_input(src)
        # _src = self.layer_norm_latent(src)


        _src, _,_ = self.self_attention(_latents, _src)

        src = self.layer_norm_latent(_latents + _src)
        src = self.feedforward(src) + src

        return src



class perceiver(nn.Module):
    def __init__(self, input_fea, num_latents, query_dim, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()
        self.num_latents = num_latents
        self.query_dim = query_dim
        self.embedding = nn.Linear(input_fea, query_dim)
        self.layers = perceiverLayer(input_fea, num_latents, query_dim, hidden_dim, n_heads, dropout_ratio, device)

        self.dropout = nn.Dropout(dropout_ratio)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.query_dim)).to(device)
        self.device = device
    def forward(self, src):

        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        
        latents = self.latents.repeat(batch_size,1,1)

        # src = self.embedding(src)

        out = self.layers(latents,src)

        
        return out