"""
    Transformer 101 > BERT 

        - this code is for educational purpose.
        - the code is written for easy understanding not for optimized code.

    Author : Sangkeun Jung (hugmanskj@gmai.com)
    All rights reserved. (2021)
"""


# In this code, we will implement
#   - BERT (Bidirectional Encoder Representations from Transformer)
#   - To implement, we re-use many parts of the pre-implemented TransformerEncoder
#   - For better understanding, 
#       - check the paramter names of BERT original implementations and those of this implementation.
#       - check how to copy huggingface parameter to this parameter


import torch
import torch.nn as nn
import torch.nn.functional as F

from commons import TransformerEncoder

# -------------------------------------------------------------------------- #
# BERT Implementation
# -------------------------------------------------------------------------- #
# BERT has three parts
#   - Embedding Part ( we will re-use huggingface code)
#       - symbol embedding
#       - position embedding
#       - type embedding
#
#   - Transformer Encoder Blocks (we will user our own code)
#
#   - Pooling Part (we will re-use huggingface code)
#
#   Note that
#       - embedding and pooling part varies a lot according to transformer researches.
# -------------------------------------------------------------------------- #


# Embedding and Pooling
# this embedding is from huggingface
class BertEmbeddings(nn.Module):
    """ this embedding moudles are from huggingface implementation
        but, it is simplified for just testing 
    """

    def __init__(self, vocab_size, hidden_size, pad_token_id, max_bert_length_size, layer_norm_eps, hidden_dropout_prob):
        super().__init__()
        self.word_embeddings        = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings    = nn.Embedding(max_bert_length_size, hidden_size)
        self.token_type_embeddings  = nn.Embedding(2, hidden_size) # why 2 ? 0 and 1 

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout   = nn.Dropout(hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(max_bert_length_size).expand((1, -1)))
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

        # always absolute
        self.position_embedding_type = "absolute"

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# this pooler is from huggingface
class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def cp_weight(src, tar, copy_bias=True, include_eps=False):
    assert tar.weight.size() == src.weight.size(), "Not compatible parameter size"
    tar.load_state_dict( src.state_dict() )
    
    if include_eps:
        # in case of LayerNorm. 
        with torch.no_grad():
            tar.eps = src.eps  

    ## call by reference
    ## therefore, tar value is changed in this func. 

def cp_encoder_block_weights_from_huggingface(src_encoder, tar_encoder):
    ## src: huggingface BERT model
    ## tar: my BERT model 
    for layer_num, src_layer in enumerate(src_encoder.layer):
        # <<< to MultiHeadAttention (wq, wk, wv, wo) >>>
        cp_weight(src_layer.attention.self.query,   tar_encoder.layers[layer_num].self_attn.wq) # wq
        cp_weight(src_layer.attention.self.key,     tar_encoder.layers[layer_num].self_attn.wk) # wk
        cp_weight(src_layer.attention.self.value,   tar_encoder.layers[layer_num].self_attn.wv) # wv
        cp_weight(src_layer.attention.output.dense, tar_encoder.layers[layer_num].self_attn.wo) # wo

        # <<< to MLP (fc1, fc2) >>>
        cp_weight(src_layer.intermediate.dense, tar_encoder.layers[layer_num].fc1) # feed_forward_1
        cp_weight(src_layer.output.dense,       tar_encoder.layers[layer_num].fc2) # feed_forward_2

        # layer normalization parameters
        cp_weight(src_layer.attention.output.LayerNorm, tar_encoder.layers[layer_num].self_attn_layer_norm, include_eps=True) # norm_1
        cp_weight(src_layer.output.LayerNorm,           tar_encoder.layers[layer_num].final_layer_norm, include_eps=True) # norm_2

    return tar_encoder


## Our Implemenation 
from argparse import ArgumentParser
from pytorch_lightning.callbacks import EarlyStopping
def cli_main():
    ## prepare huggingface BERT
    ##  - huggingface transformer is directly copyied from tensorflow's pretrained models

    ## In the below, HG stands for 'huggingface'
    from transformers import BertModel, BertTokenizer, BertConfig
    model_name = 'bert-base-cased'
    tokenizer  = BertTokenizer.from_pretrained(model_name)
    hg_bert    = BertModel.from_pretrained(model_name) ## huggingface bert
    hg_config  = BertConfig.from_pretrained(model_name)

    ## prepare my BERT 
    ## 
    ## We need to prepare and copy networks and weights 
    ##      - Embeddings
    ##      - Encoder
    ##      - Pooler
    # in case of bert-base-cased
    input_vocab_size    = tokenizer.vocab_size 
    padding_idx         = tokenizer.convert_tokens_to_ids('[PAD]')
    BERT_MAX_SEQ_LENGTH = hg_config.max_position_embeddings
    d_model             = hg_config.hidden_size

    # 1) BertBembeddings class code is from huggingface
    embeddings = BertEmbeddings(
                                        input_vocab_size,
                                        d_model,
                                        padding_idx,
                                        BERT_MAX_SEQ_LENGTH,
                                        hg_config.layer_norm_eps,     # layer norm eps
                                        hg_config.hidden_dropout_prob # 0.1    
                                )
    embeddings.load_state_dict( hg_bert.embeddings.state_dict() ) # copy parameters


    # 2) BertPooler class code is from huggingface
    pooler = BertPooler(d_model)
    pooler.load_state_dict( hg_bert.pooler.state_dict() ) # copy parameters

    # 3) Encoder Block Prepare and Parameter Copying
    encoder = TransformerEncoder(
                                        num_layers=hg_config.num_hidden_layers, 
                                        d_model=d_model, 
                                        num_heads=hg_config.num_attention_heads,
                                        dropout=hg_config.attention_probs_dropout_prob,
                                        dim_feedforward=hg_config.intermediate_size
                                )

    encoder = cp_encoder_block_weights_from_huggingface(
                                        src_encoder=hg_bert.encoder,
                                        tar_encoder=encoder 
                                      )

    input_texts =   [
                        "this is a test text", 
                        "is it working?"
                    ]
                
    tokenized_ouptut = tokenizer(input_texts, max_length=BERT_MAX_SEQ_LENGTH, padding="max_length")

    input_ids        = torch.tensor(tokenized_ouptut.input_ids)
    o_attention_mask = torch.tensor(tokenized_ouptut.attention_mask)
    token_type_ids   = torch.tensor(tokenized_ouptut.token_type_ids)

    with torch.no_grad():
        ## disable dropout -- huggingface
        hg_bert.eval() 

        ## disable dropout -- my code
        embeddings.eval() 
        pooler.eval() 
        encoder.eval() 

        ## now we need to feedforward both on huggingface BERT and our BERT
        attention_mask = o_attention_mask[:, None, None, :] # [B, 1, 1, seq_len] 

        seq_embs   = embeddings(input_ids) 
        output     = encoder(seq_embs, attention_mask)
        pooled_out = pooler(output[0]) 

        hg_output = hg_bert( 
                            input_ids=input_ids,
                            attention_mask=o_attention_mask,
                            token_type_ids=token_type_ids
                          )

        assert torch.all( torch.eq(hg_output.pooler_output, pooled_out) ), "Not same result!"

        print("\n\nSAME RESULT! -- Huggingface and My Code")

if __name__ == '__main__':
    cli_main()
