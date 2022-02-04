"""
    Transformer 101 > BERT > more Modulized version

        - this code is for educational purpose.
        - the code is written for easy understanding not for optimized code.

    Author : Sangkeun Jung (hugmanskj@gmai.com)
    All rights reserved. (2021)
"""

import torch



# In this code, we will implement
#   - BERT (Bidirectional Encoder Representations from Transformer)
#   - To implement, we re-use many parts of the pre-implemented TransformerEncoder
#   - For better understanding, 
#       - check the paramter names of BERT original implementations and those of this implementation.
#       - check how to copy huggingface parameter to this parameter
#   - All the previous things are wrapped in 'BERT' in commons.py

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

    from commons import BERT_CONFIG
    my_config = BERT_CONFIG(
                            vocab_size=tokenizer.vocab_size,
                            padding_idx=tokenizer.convert_tokens_to_ids('[PAD]'),
                            max_seq_length=hg_config.max_position_embeddings,
                            d_model=hg_config.hidden_size,
                            layer_norm_eps=hg_config.layer_norm_eps,
                            emb_hidden_dropout=hg_config.hidden_dropout_prob,
                            num_layers=hg_config.num_hidden_layers,
                            num_heads=hg_config.num_attention_heads,
                            att_prob_dropout=hg_config.attention_probs_dropout_prob,
                            dim_feedforward=hg_config.intermediate_size
                           )    
    from commons import BERT
    my_bert = BERT(my_config)
    my_bert.copy_weights_from_huggingface(hg_bert)

    
    input_texts =   [
                        "this is a test text", 
                        "is it working?"
                    ]
                
    tokenized_ouptut = tokenizer(input_texts, max_length=my_config.max_seq_length, padding="max_length")

    input_ids        = torch.tensor(tokenized_ouptut.input_ids)
    o_attention_mask = torch.tensor(tokenized_ouptut.attention_mask)
    token_type_ids   = torch.tensor(tokenized_ouptut.token_type_ids)

    with torch.no_grad():
        ## disable dropout -- huggingface
        hg_bert.eval() 
        hg_output = hg_bert( 
                            input_ids=input_ids,
                            attention_mask=o_attention_mask,
                            token_type_ids=token_type_ids
                          )

        ## disable dropout -- my code
        my_bert.eval()
        my_output, my_layer_att_scores = my_bert(input_ids=input_ids, 
                                                 token_type_ids=token_type_ids,
                                                 attention_mask=o_attention_mask)
        assert torch.all( torch.eq(hg_output.pooler_output, my_output) ), "Not same result!"

        print("\n\nSAME RESULT! -- Huggingface and My Code")

if __name__ == '__main__':
    cli_main()
