import torch
import torch.nn as nn
import math 

class InputEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_length:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_lenth = seq_length
        self.dropout = nn.Dropout(dropout)

        # Creating a Matrix of shape (seq_lenth, d_model)
        # Here seq_length is number of words in the sentence and d_model is dimension

        pe = torch.zeros(seq_length, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))
        '''
        The explanation for div term:
        Formula for positional encoding is :
        pe(pos, 2i) = sin(pos/(10000(2i/d_model))) #This is only for Even Positions
        pe(pos, 2i+1) = cos(pos/(10000^(2i/d_model))) #This is for odd positions

        Looking at the denominator term:
        1/10000^(2i/d_model) = 10000^(-d_model/2i) (inverse can be used to replace division to multiplication)
        Now, (e^ln(x) = x)
        -10000^(d_model/2i) = e^(ln(10000^(-2i/d_model)))
        => e^(ln(10000^(-2i/d_model)))
        Now, ln^(a^b) = b*ln^a
        e^(2i/d_model * ln(-10000)))
        e^(2i * (ln(-10000)/d_model))

        torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000)/d_model))
        '''

        #Apply to Even and Odd Positions
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        pe.unsqueeze(0) # (1, seq_length, d_model)
        self.register_buffer('pe', pe)

    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x-mean)/(std-self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__(self)
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        #(Batch, seq_length, d_model) -> (Batch, seq_length, d_ff)
        return self.linear_2(self.dropout(torch.relu((self.linear_1(x)))))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float):
        super().__init__(self)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.h= h
        assert d_model % h == 0, "d_model is not divisible by Head"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_score = (query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask== 0, -1e9)
        attention_score = attention_score.softmax(dim=-1)

        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score@value) , attention_score

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_score = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)


        