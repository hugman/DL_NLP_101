"""
    Transformer 101 > Decoder only architecture (KoGPT2)

        - this code is for educational purpose.
        - the code is written for easy understanding not for optimized code.

    Author : Sangkeun Jung (hugmanskj@gmai.com)
    All rights reserved. (2021)
"""


# In this code, we will implement
#   - GPT2 architecture
#       - Decoder (we will implement it from the scratch)
#   - Check carefully, How to implement
#       - remove Cross-Attention
#       - Post-LN to Pre-LN architecture !!!!
#   - Check huggingface GPT2 and our implementation shows the exact results. 


import torch
import torch.nn as nn
import torch.nn.functional as F

import copy 
import math

from commons import clones

class Conv1D(nn.Module):
    # this code is from https://amaarora.github.io/2020/02/18/annotatedGPT2.html -- and huggingface code
    # basically, it is quite ok to use Linear instead of Conv1D
    # but, to keep the consistency of original openai-gpt2 implmentation
    # we used conv1d as well.
    def __init__(self, nx, nf):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias   = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        # [B, S, dim_x] -> [B, S, dim_nf]
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

def gelu_new(x):
    """
    this code is from https://github.com/huggingface/transformers/blob/3fefa292c1c419f0c4c3e2697cdd94cafaeb4b66/src/transformers/activations.py#L37
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class GPT2MLP(nn.Module):
    # this code is from https://amaarora.github.io/2020/02/18/annotatedGPT2.html
    # renamed feedforward as MLP to follow openai-gpt2 implementation
    # compare below GPT2MLP_linear_version
    def __init__(self, d_model, nx, dropout):
        super().__init__()
        self.c_fc    = Conv1D(d_model, nx) # linear 1
        self.c_proj  = Conv1D(nx, d_model) # linear 2
        #self.act     = F.gelu
        self.act     = gelu_new  # <-- to get exact same result of huggingface. you should use it
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))

class GPT2MLP_linear_version(nn.Module):
    # CONV1D equivalent Linear implementation
    # but, you cannot import huggingface weights to this module.
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super(GPT2MLP, self).__init__()
        self.feedforward_1 = nn.Linear(d_model, dim_feedforward)
        self.act_function  = nn.GELU()
        self.feedforward_2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.feedforward_1(x)
        x = self.act_function(x)
        x = self.feedforward_2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    # from https://amaarora.github.io/2020/02/18/annotatedGPT2.html
    # but, attention mask is not fully supported. so, I modified it.
    def __init__(self, d_model, n_head, bias=True):
        super().__init__()
        self.n_head  = n_head
        self.d_model = d_model
        self.c_attn  = Conv1D(d_model, d_model*3)  # wegiht multiply part is replaced with Conv1D

        self.dropout = nn.Dropout(0.1)
        self.c_proj  = Conv1D(d_model, d_model)    # wegiht multiply part is replaced with Conv1D
        
        # We assume d_v always equals d_k
        assert d_model % n_head == 0
        self.d_k = d_model // self.n_head  # ex) d_model = 512, num_head = 8 --> d_k = 64

    def split_heads(self, x):
        new_shape = x.size()[:-1] + (self.n_head, self.d_k) 
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3) #[B, heads, seq_len, d_k]

    def _attn(self, q, k, v, mask=None):
        scores  = torch.matmul(q, k.transpose(-2, -1))
        scores  = scores/math.sqrt(v.size(-1))  # scaling by root
        nd, ns  = scores.size(-2), scores.size(-1)

        # masking 
        if mask != None:
            # sum-method
            # https://github.com/huggingface/transformers/blob/3fefa292c1c419f0c4c3e2697cdd94cafaeb4b66/src/transformers/models/gpt2/modeling_gpt2.py#L809
            mask = (1.0 - mask) * -1e4   ## follow hugging-face method  
            scores = scores + mask   

        scores  = F.softmax(scores, dim=-1) 
        scores  = self.dropout(scores)
        outputs = torch.matmul(scores, v)
        return outputs, scores
    
    def merge_heads(self, x):
        x         = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2)*x.size(-1),)
        return x.view(*new_shape)
        
    def forward(self, x, attention_mask):
        x        = self.c_attn(x) # new `x` shape - `[1,3,2304]`
        q, k, v  = x.split(self.d_model, dim=2)
        q, k, v  = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        out, scores = self._attn(q, k, v, attention_mask)
        out      = self.merge_heads(out)
        out      = self.c_proj(out)
        return out, scores


class GPT2_TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout=0.1):
        super(GPT2_TransformerBlock, self).__init__()
        self.attn        = MultiHeadAttention(d_model=d_model, n_head=n_head, bias=True)
        self.mlp         = GPT2MLP(d_model=d_model, nx=dim_feedforward, dropout=dropout)
        self.ln_1        = nn.LayerNorm(d_model)
        self.ln_2        = nn.LayerNorm(d_model)
                
    def forward(self, x, look_ahead_mask):
        # Note : PRE Layer Normalization
        # Note : attention mask for GPT2 block is only look-ahead-mask
        # 1) layernorm and masked multihead 
        nx = self.ln_1(x)
        a, attn_scores = self.attn(nx, attention_mask=look_ahead_mask) 
        x = x + a 

        # 2) layernorm and MLP
        m = self.mlp( self.ln_2(x) )
        x = x + m 
        return x, attn_scores


class GPT2Decoder(nn.Module):
    "Decoder Block of GPT2 - a stack of N layers"
    #   - the position of LayerNorm is different from original implementation
    #   - no encoder connected parts
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward=None):
        super(GPT2Decoder, self).__init__()
        self.num_layers = num_layers
        if dim_feedforward == None: dim_feedforward = 4*d_model  ## https://arxiv.org/pdf/1810.04805.pdf (page3)
        
        a_layer = GPT2_TransformerBlock(d_model=d_model, n_head=num_heads, dim_feedforward=dim_feedforward)

        # prepare N sub-blocks
        self.layers = clones(a_layer, self.num_layers)
        
    def forward(self, x, look_ahead_mask=None):
        # x : [B, tar_seq_len, d_model] 
        # enc_output : [B, src_seq_len, d_model] 
        # look_ahead_mask : 

        layers_attn_scores = []
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x, attn_scores = layer(x, look_ahead_mask)
            layers_attn_scores.append(attn_scores)

        return x, layers_attn_scores




class GPT2(nn.Module):
    """ GPT2 model """
    def __init__(self, 
                 vocab_size,    # decoder use same vocab for input and output
                 num_layers,    # number of layers
                 emb_dim,       # number embedding
                 d_model,       # dim. in attemtion mechanism 
                 num_heads,
                 max_seq_length,
                 ):
        super().__init__()
        self.max_seq_len = max_seq_length
        self.dropout_rate = 0.1 
        self.dim_feedforward = 4 * d_model  # to follow convention (transformer)

        self.tokens = 0

        # GPT INPUT PART ---------------------------------
        self.wte = nn.Embedding(vocab_size, emb_dim) # input vocab size -> emb_dim
        self.wpe = nn.Embedding(self.max_seq_len, emb_dim) # each position -> emb_dim
        self.emb_dropout = nn.Dropout(self.dropout_rate)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(self.max_seq_len).expand((1, -1)))

        # GPT TRANSFORMER PART ---------------------------
        self.blocks = GPT2Decoder(
                                        num_layers=num_layers,
                                        d_model=d_model,
                                        num_heads=num_heads,
                                        dim_feedforward=self.dim_feedforward
                                 )
        self.ln_f   = nn.LayerNorm(d_model) # to follow original gpt2 variable name

        # GPT OUTPUT PART --------------------------------
        # highgly depend on the task 
        # decoder head
        self.head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        B, seq_len = input_ids.size()
        assert seq_len <= self.max_seq_len, "Input sequence length exceed model's maximum input length"
        
        # ---- INPUT (EMBEDDING)  PART -----
        token_embeddings = self.wte(input_ids) # each index maps to a (learnable) vector
        seq_length = input_ids.shape[1]
        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.wpe(position_ids) # each position maps to a (learnable) vector
        x = self.emb_dropout(token_embeddings + position_embeddings)
        
        # ---- Transformer PART ------------
        lookahead_mask = self.look_ahead_mask(seq_len).to(x.device) # mask : head compatible form.
        x, layer_attn_scores = self.blocks(x, look_ahead_mask=lookahead_mask)
        x = self.ln_f(x)  # <-- layer norm on the final transformer block

        # --- OUTPUT PART ------------------
        logits = self.head(x)

        return logits

    def look_ahead_mask(self, tgt_len:int) -> torch.FloatTensor:  
        mask = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.int), diagonal=1)
        mask = 1 - mask # reverse
        return mask
        
    

def cp_weight(src, tar, copy_bias=True, include_eps=False):
    assert tar.weight.size() == src.weight.size(), "Not compatible parameter size"
    tar.load_state_dict( src.state_dict() )
    
    if include_eps:
        # in case of LayerNorm. 
        with torch.no_grad():
            tar.eps = src.eps  

    ## call by reference
    ## therefore, tar value is changed in this func. 

def cp_gpt2_transformer_block_weights(src, tar):
    ## src: huggingface GPT2 - Transformer model 
    ## tar: my GPT2 - model - core weights

    ## layer normalization at top transformer block 
    cp_weight(src.transformer.ln_f, tar.ln_f, include_eps=True) # ln_f

    ## layer weights
    for layer_num, src_block in enumerate(src.transformer.h):
        # <<< MultiHeadAttention (Conv1D's parameters) >>>
        cp_weight(src_block.attn.c_attn,        tar.blocks.layers[layer_num].attn.c_attn) # c_attn
        cp_weight(src_block.attn.c_proj,        tar.blocks.layers[layer_num].attn.c_proj) # c_proj

        # same dropout for attention, residual, and others
        #tar.blocks.layers[layer_num].attn.dropout.load_state_dict( src_block.attn.attn_dropout )

        # <<< MLP >>
        cp_weight(src_block.mlp.c_fc,       tar.blocks.layers[layer_num].mlp.c_fc) # c_fc
        cp_weight(src_block.mlp.c_proj,     tar.blocks.layers[layer_num].mlp.c_proj) # c_proj
        #tar.blocks.layers[layer_num].mlp.dropout.load_state_dict(src_block.mlp.dropout) # dropout

        # layer normalization parameters
        cp_weight(src_block.ln_1, tar.blocks.layers[layer_num].ln_1, include_eps=True) # ln_1
        cp_weight(src_block.ln_2, tar.blocks.layers[layer_num].ln_2, include_eps=True) # ln_2

    return tar


## Our Implemenation 
from argparse import ArgumentParser
from pytorch_lightning.callbacks import EarlyStopping
def cli_main():
    # -------------- GPT2 model -------------- ##
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
                        "skt/kogpt2-base-v2",
                        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                        pad_token='<pad>', mask_token='<mask>'
                    ) 

    from transformers import GPT2LMHeadModel
    inputs = tokenizer("가을 하늘은 참 맑습니다~", return_tensors="pt")

    ## huggingface model
    hg_model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

    ## my model
    my_model = GPT2(
                        vocab_size=hg_model.config.vocab_size, 
                        num_layers=hg_model.config.n_layer,
                        emb_dim=hg_model.config.n_embd,
                        d_model=hg_model.config.n_embd,
                        num_heads=hg_model.config.n_head,
                        max_seq_length=hg_model.config.n_ctx,
                    )

    ## [INPUT EMBEDDING] 
    ## copy embeddings from huggingface to my gpt2
    my_model.wte.load_state_dict( hg_model.transformer.wte.state_dict() )
    my_model.wpe.load_state_dict( hg_model.transformer.wpe.state_dict() )

    ## [OUTPUT EMBEDDING]
    ## copy to output vocab
    my_model.head.load_state_dict( hg_model.lm_head.state_dict() )

    ## [TRANSFORMER BLOCK]
    ## transformer block copy 
    my_model = cp_gpt2_transformer_block_weights(hg_model, my_model)

    hg_model.eval()
    my_model.eval()

    with torch.no_grad():
        hg_outputs = hg_model(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask
                        )
        my_output = my_model(
                            input_ids=inputs.input_ids,
                            #attention_mask=inputs.attention_mask  # <-- we don't need padding mask for GPT1, GPT2
                        )
        assert torch.all( torch.eq(hg_outputs.logits, my_output) ), "Not same result!"

        print("\n\nSAME RESULT! -- Huggingface-KoGPT2 and My Code")


if __name__ == '__main__':
    cli_main()