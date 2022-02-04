"""
    Transformer 101/Attention 101 > Dot-Product + Query, Key, Value + Multihead Attention

        - this code is for educational purpose.
        - the code is written for easy understanding not for optimized code.

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
    d_k  = k.size()[-1]
    #attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) # [B, num_heads, query_seq_len, key_seq_len] 
    
    attn = torch.matmul(q, k.transpose(-2, -1))


    # masking 
    if mask != None:
        inverted_mask = 1.0 - mask
        inverted_mask = inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(attn.dtype).min)
        
        # broadcasting will be happened here
        # [B, N, len_query, len_key] + [B, 1, 1, len_key] --> 
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


## ------------------------------------------------------------------------ ##
## Training and Testing with toy dataset                                    ##
## ------------------------------------------------------------------------ ##
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np 

def load_data(fn):
    data = []
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()

            seq_str, query, y = line.split('\t')
            seqs = seq_str.split(',')
            data.append( (seqs, query, y) )
    return data

# you can define any type of dataset
# dataset : return an example for batch construction. 
class NumberDataset(Dataset):
    """Dataset."""

    def __init__(self, fn, input_vocab, output_vocab, max_seq_length):
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.max_seq_length = max_seq_length 
        
        # load 
        self.data = load_data(fn)

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx): 
        seq, q, y = self.data[idx]

        # [ input ]
        seq_ids = [ self.input_vocab[t] for t in seq ]

        # <pad> processing
        pad_id      = self.input_vocab['<pad>']
        num_to_fill = self.max_seq_length - len(seq)
        seq_ids     = seq_ids + [pad_id]*num_to_fill

        # mask processing (1 for valid, 0 for invalid)
        weights = [1]*len(seq) + [0]*num_to_fill

        # [ query ]
        # NOTE : we assume that query vocab space is same as input vocab space

        q_id = self.input_vocab[q] # enable valid query 
        #q_id = 0 # disable query -- to check query effect in attention mechanism

        # [ ouput ] 
        y_id = self.output_vocab[y]

        item = [
                    # input
                    np.array(seq_ids),
                    q_id,
                    np.array(weights),

                    # output
                    y_id
               ]
        return item 


class NumberDataModule(pl.LightningDataModule):
    def __init__(self, 
                 max_seq_length: int=12,
                 batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length 

        input_vocab, output_vocab = self.make_vocab('./data/numbers/train.txt')
        self.input_vocab_size = len( input_vocab )
        self.output_vocab_size = len( output_vocab )
        self.padding_idx = input_vocab['<pad>']

        self.input_r_vocab  = { v:k for k,v in input_vocab.items() }
        self.output_r_vocab = { v:k for k,v in output_vocab.items() }

        self.all_train_dataset = NumberDataset('./data/numbers/train.txt', input_vocab, output_vocab, max_seq_length)
        self.test_dataset      = NumberDataset('./data/numbers/test.txt', input_vocab, output_vocab, max_seq_length)

        # random split train / valiid for early stopping
        N = len(self.all_train_dataset)
        tr = int(N*0.8) # 8 for the training
        va = N - tr     # 2 for the validation 
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.all_train_dataset, [tr, va])

    def make_vocab(self, fn):
        input_tokens = []
        output_tokens = []
        data = load_data(fn)

        for seqs, query, y in data:
            for token in seqs:
                input_tokens.append(token)
            output_tokens.append(y)
        
        input_tokens = list(set(input_tokens))
        output_tokens = list(set(output_tokens)) 

        input_tokens.sort()
        output_tokens.sort()

        # [input vocab]
        # add <pad> symbol to input tokens as a first item
        input_tokens = ['<pad>'] + input_tokens 
        input_vocab = { str(token):index for index, token in enumerate(input_tokens) }

        # [output voab]
        output_vocab = { str(token):index for index, token in enumerate(output_tokens) }

        return input_vocab, output_vocab

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True) # NOTE : Shuffle

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

from torchmetrics import functional as FM

class Attention_Number_Finder(pl.LightningModule): 
    def __init__(self, 
                 # network setting
                 input_vocab_size,
                 output_vocab_size,
                 d_model,      # dim. in attemtion mechanism 
                 num_heads,    # number of heads,
                 padding_idx,
                 # optiimzer setting
                 learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()  


        # note 
        #   - the dimension for query and multi-items do not need to be same. 
        #   - for simplicity, we make all the dimensions as same. 

        # symbol_number_character to vector_number
        self.digit_emb = nn.Embedding(self.hparams.input_vocab_size, 
                                      self.hparams.d_model, 
                                      padding_idx=self.hparams.padding_idx)

        # sequence encoder using RNN
        self.encoder = nn.LSTM(d_model, int(self.hparams.d_model/2), # why? since bidirectional LSTM
                            num_layers=2, 
                            bidirectional=True,
                            batch_first=True
                          )

        # attention mechanism
        self.att = Attention(
                               d_model=self.hparams.d_model,
                               num_heads=self.hparams.num_heads
                             )

        # [to output]
        self.to_output = nn.Linear(self.hparams.d_model, self.hparams.output_vocab_size) # D -> a single number

        # loss
        self.criterion = nn.CrossEntropyLoss()  

    def forward(self, seq_ids, q_id, weight):
        # ------------------- ENCODING with ATTENTION -----------------#
        # [ Digit Character Embedding ]
        # seq_ids : [B, max_seq_len]
        seq_embs = self.digit_emb(seq_ids.long()) # [B, max_seq_len, emb_dim]

        # [ Sequence of Numbers Encoding ]
        seq_encs, _ = self.encoder(seq_embs) # [B, max_seq_len, enc_dim*2]  since we have 2 layers
        
        # with query (context)
        query = self.digit_emb(q_id) # [B, emb_dim=d_model]
        query = query.unsqueeze(1)   # [B, 1, d_model]  <- dummy dimension 

        # dynamic encoding-summarization (blending)
        multiple_items = seq_encs    # [B, max_seq_len, d_model]

        # masking - shape change
        #   mask always applied to the last dimension explicitly. 
        #   so, we need to prepare good shape of mask
        #   to prepare [B, dummy_for_heads, dummy_for_query, dim_for_key_dimension]
        mask = weight[:, None, None, :] # [B, 1, 1, max_seq_len]

        blendded_vector, attention_scores = self.att(query=query, 
                                                     key=multiple_items,
                                                     value=multiple_items, 
                                                     mask=mask) 
        # blendded_vector  : [B, query_seq_len=1, d_model]
        # attention_scores : [B, num_heads, query_seq_len=1, max_seq_len] 
        blendded_vector = blendded_vector.squeeze(1) # since we use a single query 
        
        # To output
        logits = self.to_output(blendded_vector)
        return logits, attention_scores

    def training_step(self, batch, batch_idx):
        seq_ids, q_id, weights, y_id = batch 
        logits, _ = self(seq_ids, q_id, weights)  # [B, output_vocab_size]
        loss = self.criterion(logits, y_id.long()) 
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # all logs are automatically stored for tensorboard
        return loss

    def validation_step(self, batch, batch_idx):
        seq_ids, q_id, weights, y_id = batch 

        logits, _ = self(seq_ids, q_id, weights)  # [B, output_vocab_size]
        loss = self.criterion(logits, y_id.long()) 
        
        ## get predicted result
        prob = F.softmax(logits, dim=-1)
        acc = FM.accuracy(prob, y_id)
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def validation_step_end(self, val_step_outputs):
        val_acc  = val_step_outputs['val_acc'].cpu()
        val_loss = val_step_outputs['val_loss'].cpu()

        self.log('validation_acc',  val_acc, prog_bar=True)
        self.log('validation_loss', val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        seq_ids, q_id, weights, y_id = batch 

        logits, _ = self(seq_ids, q_id, weights)  # [B, output_vocab_size]
        loss = self.criterion(logits, y_id.long()) 
        
        ## get predicted result
        prob = F.softmax(logits, dim=-1)
        acc = FM.accuracy(prob, y_id)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics, on_epoch=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ATTENTION")
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parent_parser



from argparse import ArgumentParser
from pytorch_lightning.callbacks import EarlyStopping
def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--d_model',    default=512, type=int)  # dim. for attention model 
    parser.add_argument('--num_heads',  default=8,   type=int)  # number of attention heads

    parser = pl.Trainer.add_argparse_args(parser)
    parser = Attention_Number_Finder.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = NumberDataModule.from_argparse_args(args)
    iter(dm.train_dataloader()).next() # <for testing 


    # ------------
    # model
    # ------------
    model = Attention_Number_Finder(dm.input_vocab_size,
                                    dm.output_vocab_size,
                                    args.d_model,       # dim. in attemtion mechanism 
                                    args.num_heads,     # number of heads in attention 
                                    dm.padding_idx,
                                    args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
                            max_epochs=6, 
                            callbacks=[EarlyStopping(monitor='val_loss')],
                            gpus = 1 # if you have gpu -- set number, otherwise zero
                        )
    trainer.fit(model, datamodule=dm)

    # ------------
    # testing
    # ------------
    result = trainer.test(model, test_dataloaders=dm.test_dataloader())
    print(result)

    # {'test_acc': 0.9998000264167786, 'test_loss': 0.0018601451301947236}


if __name__ == '__main__':
    cli_main()