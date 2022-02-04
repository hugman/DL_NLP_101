"""
    BERT Related codes and modules
    
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


from commons.base import TransformerEncoder, cp_weight

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

class BERT_CONFIG():
    def __init__(self, vocab_size, padding_idx, max_seq_length, 
                       d_model, layer_norm_eps, emb_hidden_dropout,
                       num_layers, num_heads, att_prob_dropout, dim_feedforward
                 ):
        # embedding
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.layer_norm_eps = layer_norm_eps
        self.emb_hidden_dropout = emb_hidden_dropout

        # attention
        self.num_layers=num_layers
        self.num_heads = num_heads
        self.att_prob_dropout = att_prob_dropout
        self.dim_feedforward = dim_feedforward

    def __init__(self, hg_config):
        # embedding
        self.vocab_size = hg_config.vocab_size
        self.padding_idx = hg_config.pad_token_id
        self.max_seq_length = hg_config.max_position_embeddings
        self.d_model = hg_config.hidden_size
        self.layer_norm_eps = hg_config.layer_norm_eps
        self.emb_hidden_dropout = hg_config.hidden_dropout_prob

        # attention
        self.num_layers= hg_config.num_hidden_layers
        self.num_heads = hg_config.num_attention_heads
        self.att_prob_dropout = hg_config.attention_probs_dropout_prob
        self.dim_feedforward = hg_config.intermediate_size


## We will wrap-all BERT sub-processing as BERT module
class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()

        ## [Embeddings] 
        self.embeddings = BertEmbeddings(
                                            config.vocab_size ,
                                            config.d_model,
                                            config.padding_idx,
                                            config.max_seq_length,
                                            config.layer_norm_eps,    # layer norm eps
                                            config.emb_hidden_dropout # 0.1    
                                    )
        ## [Transformers]
        self.encoder = TransformerEncoder(
                                        num_layers=config.num_layers, 
                                        d_model=config.d_model, 
                                        num_heads=config.num_heads,
                                        dropout=config.att_prob_dropout,
                                        dim_feedforward=config.dim_feedforward
                                )


        ## [Pooler]
        self.pooler = BertPooler(config.d_model)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        attention_mask = attention_mask[:, None, None, :] # [B, 1, 1, seq_len] 
        seq_embs   = self.embeddings(input_ids, token_type_ids)
        output     = self.encoder(seq_embs, attention_mask)
        pooled_out = self.pooler(output[0]) 

        seq_hidden_states = output[0]
        layer_attention_scores = output[1]
        return pooled_out, seq_hidden_states, layer_attention_scores

    def cp_encoder_block_weights_from_huggingface(self, src_encoder, tar_encoder):
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

    def copy_weights_from_huggingface(self, hg_bert):
        self.embeddings.load_state_dict( hg_bert.embeddings.state_dict() ) 
        self.pooler.load_state_dict( hg_bert.pooler.state_dict() ) 

        self.encoder = self.cp_encoder_block_weights_from_huggingface(
                                            src_encoder=hg_bert.encoder,
                                            tar_encoder=self.encoder 
                                          )


