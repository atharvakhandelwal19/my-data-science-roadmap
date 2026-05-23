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

