"""
    Transformer 101 > Encoder Part Implementation

        - this code is for educational purpose.
        - the code is written for easy understanding not for optimized code.

    Author : Sangkeun Jung (hugmanskj@gmai.com)
    All rights reserved. (2021)
"""


# In this code, we will implement
#   - Transformer Encoder Part (only single layer)
#   - Especially, we will focus on 
#       - Residual Connection
#       - Layer Normalization
#       - Scaling 
#   - We re-use and wrapped pre-implemented Dot-QKV-Multihead Attention process
#   - By doing this, we will remove 'LSTM' part for sequence encoding

import torch
import torch.nn as nn
import torch.nn.functional as F

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

            query_item_seq_str, y = line.split('\t')
            all_tokens = query_item_seq_str.split(',')
            q_tokens = all_tokens[0].split('|')
            i_tokens = all_tokens[1:]

            tokens = [q_tokens[0], '|'] + [q_tokens[1]] + i_tokens 
            data.append( (tokens, y) )
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
        seq, y = self.data[idx]

        # [ input ]
        seq_ids = [ self.input_vocab[t] for t in seq ]

        # <pad> processing
        pad_id      = self.input_vocab['<pad>']
        num_to_fill = self.max_seq_length - len(seq)
        seq_ids     = seq_ids + [pad_id]*num_to_fill

        # mask processing (1 for valid, 0 for invalid)
        weights = [1]*len(seq) + [0]*num_to_fill

        # [ ouput ] 
        y_id = self.output_vocab[y]

        item = [
                    # input
                    np.array(seq_ids),
                    np.array(weights),

                    # output
                    y_id
               ]
        return item 


class NumberDataModule(pl.LightningDataModule):
    def __init__(self, 
                 max_seq_length: int=15,
                 batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length 

        input_vocab, output_vocab = self.make_vocab('./data/numbers/train.seq.txt')
        self.input_vocab_size = len( input_vocab )
        self.output_vocab_size = len( output_vocab )
        self.padding_idx = input_vocab['<pad>']

        self.input_r_vocab  = { v:k for k,v in input_vocab.items() }
        self.output_r_vocab = { v:k for k,v in output_vocab.items() }

        self.all_train_dataset = NumberDataset('./data/numbers/train.seq.txt', input_vocab, output_vocab, max_seq_length)
        self.test_dataset      = NumberDataset('./data/numbers/test.seq.txt', input_vocab, output_vocab, max_seq_length)

        # random split train / valiid for early stopping
        N = len(self.all_train_dataset)
        tr = int(N*0.8) # 8 for the training
        va = N - tr     # 2 for the validation 
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.all_train_dataset, [tr, va])

    def make_vocab(self, fn):
        input_tokens = []
        output_tokens = []
        data = load_data(fn)

        for tokens, y in data:
            for token in tokens:
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


## reused 
from attentions import Attention

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


class TransformerEncoder_Number_Finder(pl.LightningModule): 
    def __init__(self, 
                 # network setting
                 input_vocab_size,
                 output_vocab_size,
                 d_model,      # dim. in attemtion mechanism 
                 num_heads,    # number of heads
                 padding_idx,
                 # optiimzer setting
                 learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()  

        # symbol_number_character to vector_number
        self.input_emb = nn.Embedding(self.hparams.input_vocab_size, 
                                      self.hparams.d_model, 
                                      padding_idx=self.hparams.padding_idx)

        # Now, we use transformer-encoder for encoding
        #   - multiple items and a query item together
        self.encoder = TransformerEncoderLayer( self.hparams.d_model, 
                                                self.hparams.num_heads, 
                                                dim_feedforward=self.hparams.d_model*4, # by convention
                                                dropout=0.1
                                              )


        # [to output]
        self.to_output = nn.Linear(self.hparams.d_model, self.hparams.output_vocab_size) # D -> a single number

        # loss
        self.criterion = nn.CrossEntropyLoss()  

    def forward(self, seq_ids, weight):
        # INPUT EMBEDDING
        # [ Digit Character Embedding ]
        # seq_ids : [B, max_seq_len]
        seq_embs = self.input_emb(seq_ids.long()) # [B, max_seq_len, d_model]

        # ENCODING BY Transformer-Encoder
        # [mask shaping]
        # masking - shape change
        #   mask always applied to the last dimension explicitly. 
        #   so, we need to prepare good shape of mask
        #   to prepare [B, dummy_for_heads, dummy_for_query, dim_for_key_dimension]
        mask = weight[:, None, None, :] # [B, 1, 1, max_seq_len]
        seq_encs, attention_scores = self.encoder(seq_embs, mask) # [B, max_seq_len, d_model] 

        # seq_encs         : [B, max_seq_len, d_model]
        # attention_scores : [B, max_seq_len_query, max_seq_len_key]

        # Output Processing
        # pooling 
        blendded_vector = seq_encs[:,0]  # taking the first(query) - step hidden state
        
        # To output
        logits = self.to_output(blendded_vector)
        return logits, attention_scores

    def training_step(self, batch, batch_idx):
        seq_ids, weights, y_id = batch 
        logits, _ = self(seq_ids, weights)  # [B, output_vocab_size]
        loss = self.criterion(logits, y_id.long()) 
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # all logs are automatically stored for tensorboard
        return loss

    def validation_step(self, batch, batch_idx):
        seq_ids, weights, y_id = batch 

        logits, _ = self(seq_ids, weights)  # [B, output_vocab_size]
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
        seq_ids, weights, y_id = batch 

        logits, _ = self(seq_ids, weights)  # [B, output_vocab_size]
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
    parser.add_argument('--num_heads',  default=8, type=int)    # number of multi-heads

    parser = pl.Trainer.add_argparse_args(parser)
    parser = TransformerEncoder_Number_Finder.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = NumberDataModule.from_argparse_args(args)
    iter(dm.train_dataloader()).next() # <for testing 


    # ------------
    # model
    # ------------
    model = TransformerEncoder_Number_Finder(
                                                dm.input_vocab_size,
                                                dm.output_vocab_size,
                                                args.d_model,       # dim. in attemtion mechanism 
                                                args.num_heads,
                                                dm.padding_idx,
                                                args.learning_rate
                                            )

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
                            max_epochs=2, 
                            callbacks=[EarlyStopping(monitor='val_loss')],
                            gpus = 1 # if you have gpu -- set number, otherwise zero
                        )
    trainer.fit(model, datamodule=dm)

    # ------------
    # testing
    # ------------
    result = trainer.test(model, test_dataloaders=dm.test_dataloader())
    print(result)

    # {'test_acc': 0.9998000264167786, 'test_loss': 0.0011931280605494976}
   

if __name__ == '__main__':
    cli_main()