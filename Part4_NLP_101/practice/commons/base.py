"""
    Attention Related codes and modules
    
    Author : Sangkeun Jung (hugmanskj@gmai.com)
    All rights reserved. (2021)
"""


# In this code, we will implement
#   - Scaled Dot-Product attention mechanism 
#   - Query Key Value attention 
#   - Multihead attention



import torch
import torch.nn as nn
import torch.nn.functional as F

import math 
def scaled_dot_product_attention(   q: torch.Tensor, 
                                    k: torch.Tensor, 
                                    v: torch.Tensor,                                  
                                    mask: torch.Tensor = None,
                                    dropout: float = 0.1,
                                 ) -> torch.Tensor:
    """
        In here, we try to calculate all multi-heads attentions at once. 
        So, we assumed that the first dimension of q, k and v is B*num_heads=...
            q : expect [..., query_seq_len, d_k]
            k : expect [..., key_seq_len,   d_k]
            v : expect [..., key_seq_len,   d_v]
        mask : expect extended shaped [B, num_heads, query_seq_len, key_seq_len] 1.0 for attend elements, 0 for masking elements
        dropout : expect float value. 
    """
    # for scaling
    d_k = k.size()[-1]
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) # [B, num_heads, query_seq_len, key_seq_len] 

    # masking 
    if mask != None:
        inverted_mask = 1.0 - mask
        inverted_mask = inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(attn.dtype).min)
        attn = attn + inverted_mask  # checkout before and after attn[0][0][0], mask[0][0][0]

    # calculate softmax 
    attention_weights = F.softmax(attn, dim=-1)  # over key dimension   # [..., seq_len, d_k]

    # Original Paper didn't mention about dropout on attention weights. 
    # But many working architectures use dropout on attentions 
    # so, in here we will apply dropout on scores
    if type(dropout) == float : 
        attention_weights = F.dropout(attention_weights, dropout)
    else: 
        attention_weights = dropout(attention_weights)

    # blending
    output = torch.matmul(attention_weights, v)
    return output, attention_weights

class Attention(nn.Module):
    ## this Attention implementation is almost identical to original transformer paper.
    def __init__(self, d_model, num_heads, dropout=0.1, use_bias=True):
        super(Attention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads

        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads  # ex) d_model = 512, num_head = 8 --> d_k = 64
        self.d_v = d_model // num_heads  # ex) d_model = 512, num_head = 8 --> d_v = 64

        # why * num_head? --> preapre N heads's input
        # d_model = self.d_k * self.num_head
        # 
        # there are variations to use 'biases' in q,k,v, and o 
        # but, in this implementation, we will use bias 
        self.wq = nn.Linear(d_model, d_model, bias=use_bias) 
        self.wk = nn.Linear(d_model, d_model, bias=use_bias) 
        self.wv = nn.Linear(d_model, d_model, bias=use_bias) 

        # dropout
        self.dropout = nn.Dropout(dropout)

        # to make output 
        # we follow original transformer paper : 
        # in the paper, they mentioned WO for projection on concat vector.
        self.wo = nn.Linear(d_model, d_model, bias=use_bias)

    def split_heads(self, x, batch_size):
        # split the projected dimension 
        # [B, seq_len, heads * d_k ] --> [B, heads, seq_len, d_k] 
        x = x.view(batch_size, -1, self.num_heads, self.d_k) # to be [B, seq_len, heads, d_k]
        x = x.transpose(1,2).contiguous()  # to be [B, heads, seq_len, d_k]
        return x

    def forward(self, query, key, value, mask=None):
        q = self.wq(query)      # d_k --> d_k*num_head
        k = self.wk(key)        # d_k --> d_k*num_head
        v = self.wv(value)      # d_k --> d_k*num_head
        
        # shape change to [B, heads, seq_len, d_k]
        _, qS = q.size()[0], q.size()[1] # qS = query_seq_len 
        B, S  = k.size()[0], k.size()[1] # S  = key_seq_len
        
        q = self.split_heads(q, B) # [B, num_heads, query_seq_len, d_k]
        k = self.split_heads(k, B) # [B, num_heads, key_seq_len,   d_k]
        v = self.split_heads(v, B) # [B, num_heads, key_seq_len,   d_k]

        # scaled dot-product attention
        # scaled_attention  = [..., query_seq_len, d_k]
        # attention_weights = [..., query_seq_len, key_seq_len]
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        
        # [Concat Process - for merging multiheads] 
        # recover the tensor form
        scaled_attention = scaled_attention.transpose(1,2)     # to be [B, query_seq_len, num_heads, d_k]
        
        # concat
        concat_attention = scaled_attention.reshape(B, qS, -1) # to be [B, query_seq_len, (num_heads*d_k)=d_model]

        # to output
        output = self.wo(concat_attention) 

        # output : # [B, query_seq_len, d_model]
        # attention_weights : [B, num_heads, query_seq_len, key_seq_len]
        return output, attention_weights 


class TransformerEncoderLayer(nn.Module):
    # - a single layer for Transformer-Encoder block
    # - This Encoder block is almost identical to original transformer block
    # - activation function is changed to RELU 
    #       - (note that, recently RELU is frequently replaced as GELU)

    def __init__(self, d_model, num_head, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.dropout = dropout

        # self-attention
        self.self_attn = Attention(d_model, num_head, dropout)

        # MLP
        self.act_fc = nn.GELU() # <- I changed RELU to GELU 
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)

        # LN for after attention and final 
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.final_layer_norm     = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # 1) self-multihead-attention with add & norm 
        residual = x
        x, attn_scores = self.self_attn(query=x, key=x, value=x, mask=mask)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x 
        x = self.self_attn_layer_norm(x) # POST Layer Normalization

        # 2) MLP with add & norm
        residual = x
        x = self.act_fc(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x 
        x = self.final_layer_norm(x)     # POST Layer Normalization

        # out : [batch_size, step_size=S, d_model]
        return x, attn_scores

   
import copy 
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerEncoder(nn.Module):
    # Encoder Block - a stack of N layers
    # Exactly same as TransformerEncoder 
    def __init__(self, num_layers, d_model, num_heads, dropout, dim_feedforward=None):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        if dim_feedforward == None: dim_feedforward = 4*d_model  ## https://arxiv.org/pdf/1810.04805.pdf (page3)
        
        a_layer = TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)

        # prepare N sub-blocks
        self.layers = clones(a_layer, self.num_layers)
        
    def forward(self, x, mask=None):
        # x expects : [B, seq_len, d_model] 
        layers_attn_scores = []

        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x, attn_scores = layer(x, mask)
            layers_attn_scores.append(attn_scores)
        return x, layers_attn_scores


def cp_weight(src, tar, copy_bias=True, include_eps=False):
    assert tar.weight.size() == src.weight.size(), "Not compatible parameter size"
    tar.load_state_dict( src.state_dict() )
    
    if include_eps:
        # in case of LayerNorm. 
        with torch.no_grad():
            tar.eps = src.eps  

    ## call by reference
    ## therefore, tar value is changed in this func. 
      


## ---------------- DECODER ----------------------- ##
class TransformerDecoderLayer(nn.Module):
    # - a single layer for Transformer-Decoder block
    def __init__(self, d_model, num_head, droput, dim_feedforward, eps=1e-12):
        super(TransformerDecoderLayer, self).__init__()
        self.embed_dim = d_model 
        self.dropout = droput

        ## self-attention
        self.self_attn = Attention(self.embed_dim, num_head, droput)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim) ## residual + LN

        ## cross-attention over encoder's output
        self.encoder_attn = Attention(self.embed_dim, num_head, droput)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=eps)

        ## MLP
        self.act_fc = nn.GELU()
        self.activation_dropout = droput # same as hidden state dropout
        
        self.fc1 = nn.Linear(self.embed_dim, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=eps)
      

    def forward(self, x, enc_output, look_ahead_mask, enc_pad_mask):
        "Follow Figure 1 (right) of the original paper for connections."
        # enc_output : [B, input_seq_len, d_model]
        # x : input 
        # look_ahead_mask : for decoder's input
        # enc pad_mask    : for encoder's output

        # 1) self-multihead-attention with add & norm 
        residual = x
        x, dec_attn_scores = self.self_attn(query=x, key=x, value=x, mask=look_ahead_mask)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x 
        x = self.self_attn_layer_norm(x)

        # 2) cross attention 
        residual = x
        x, cross_attn_scores = self.encoder_attn(query=x, key=enc_output, value=enc_output, mask=enc_pad_mask)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x 
        x = self.encoder_attn_layer_norm(x)

        # 3) MLP
        residual = x
        x = self.act_fc(self.fc1(x))
        x = F.dropout(x, self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x 
        x = self.final_layer_norm(x)
        
        # out : [batch_size, target_seq_len, d_model]
        # attn_scores_1 : [batch_size, num_head, target_seq_len, target_seq_len] = [B, H, query_len, query_len]
        # attn_scores_2 : [batch_size, num_head, target_seq_len, source_seq_len] = [B, H, key_len, query_len]
        return x, dec_attn_scores, cross_attn_scores

class TransformerDecoder(nn.Module):
    "Decoder Block - a stack of N layers"
    def __init__(self, num_layers, d_model, num_heads, dropout, dim_feedforward=None):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers

        if dim_feedforward == None: dim_feedforward = 4*d_model
        a_layer = TransformerDecoderLayer(d_model, num_heads, dropout, dim_feedforward)

        # prepare N sub-blocks
        self.layers = clones(a_layer, self.num_layers)
        
    def forward(self, x, enc_output, look_ahead_mask=None, enc_pad_mask=None):
        # x : [B, tar_seq_len, d_model] 
        # enc_output : [B, src_seq_len, d_model] 
        # look_ahead_mask : for decoding (causual mask)
        # enc_pad_mask : for blending encoder's hidden states(key) with decoder's input(query), 
        #                need to ignore 'pad' positioned hidden states.

        layers_attn_scores_1 = []
        layers_attn_scores_2 = []

        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x, attn_scores_1, attn_scores_2 = layer(x, enc_output, look_ahead_mask, enc_pad_mask)
            layers_attn_scores_1.append(attn_scores_1) # for encoder
            layers_attn_scores_2.append(attn_scores_2) # for decoder

        return x, layers_attn_scores_1, layers_attn_scores_2



## ----------------------- tools ------------------------- ##
def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


