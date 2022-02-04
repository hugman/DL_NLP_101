"""
    Transformer 101 > Encoder + Decoder

        - this code is for educational purpose.
        - the code is written for easy understanding not for optimized code.

    Author : Sangkeun Jung (hugmanskj@gmai.com)
    All rights reserved. (2021)
"""


# In this code, we will implement
#   - Original Transformer 
#       - Encoder (we will re-use encoder implementations)
#       - Decoder (we will implement it from the scratch)
#   - Check carefully, How to implement
#       - Cross-Attention (for giving encoder info. to decoder)
#       - Look-ahead Masking (for ignoring future-information)
#   - Also note that
#       - encoder sequence length might not same as decoder sequence length
#       - carefully, check query length and key length
#
#   - For the test dataset, 
#       - We will use number sorting dataset 
#       - Generate ordered sequences of numbers removing duplicated numbers



import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from transformers.models.bert.tokenization_bert import VOCAB_FILES_NAMES 

def result_collapse(outputs, target):
    if len( [x[target] for x in outputs][0].shape ) == 0:
        target_value = torch.stack([x[target] for x in outputs])
    else:
        target_value = torch.cat([x[target] for x in outputs])
    return target_value


from commons import Attention, clones
from commons import TransformerEncoder

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

    
## -------------------- TRANSFORMER (Encoder + Decoder) ----------------------- ##
##  Additionally we need to implement
##      - Embedding modules ( we will re-use BertEmbeddings ) with differnt name
##          - wihtout token type embedding
class InputEmbeddings(nn.Module):
    """ this embedding moudles are from huggingface implementation
        but, it is simplified -- removing token type embedding since it is not for BERT
    """
    def __init__(self, vocab_size, hidden_size, pad_token_id, max_length_size, layer_norm_eps, hidden_dropout_prob):
        super().__init__()
        self.word_embeddings        = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings    = nn.Embedding(max_length_size, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout   = nn.Dropout(hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(max_length_size).expand((1, -1)))
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

        # always absolute
        self.position_embedding_type = "absolute"

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds 
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings



class Transformer(nn.Module):
    # in here, embedding and decoder output processing parts are not included
    def __init__(self, num_layers, d_model, num_heads, dropout, dim_feedforward=None):
        super().__init__()

        ## transformer blocks only 
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dropout, dim_feedforward)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, dropout, dim_feedforward)

    def create_padding_mask(self, mask):
        # prepare padding mask for attention matrix compatible
        return mask[:, None, None, :] # [B, 1, 1, seq_len] 

    def create_look_ahead_mask(self, seq_len):  
        """
        prepare causual mask or look-ahead-mask for the decoding
        In decoder, self-attention should be performed with only visible items
        at each time steps. This mask is for preventing future-items at each self-attention in decoer 
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.int), diagonal=1)
        mask = 1 - mask # reverse (1 for visible, 0 for invisible)
        return mask

    def forward(self, enc_input, dec_input, enc_pad_mask):
        # enc_input : [B, src_len, d_model]
        # dec_input : [B, tar_len, d_model]
        #               - in training, it is a right-shifted decoder output starting with <start>
        #               - in inference, it is a previous decoder output appended data starting with <start>
        #
        # enc_pad_mask : 
        #       - padding mask for encoder attention
        #       - padding mask for decoder's 2nd attention (to blend encoder's outputs)

        # --------
        # encoder
        # --------
        enc_pad_mask = self.create_padding_mask(enc_pad_mask)
        enc_output, enc_layer_att_scores = self.encoder(enc_input, enc_pad_mask)

        # --------
        # decoder
        # --------
        # masking for self-attention in decoder (LOOK-AHEAD)
        dec_length = dec_input.size()[1] 
        look_ahead_mask = self.create_look_ahead_mask(dec_length).to(dec_input.device)

        # masking for cross-attention in bleding decoder input(query) with encoder output(key, value)
        # since multiple-items are from encoder, 
        # the mask should be encoder padding mask
        dec_output, dec_layer_att_scores, dec_layer_cross_att_scores = self.decoder(
                                                                                    dec_input, 
                                                                                    enc_output, 
                                                                                    look_ahead_mask=look_ahead_mask, 
                                                                                    enc_pad_mask=enc_pad_mask
                                                                                )
        return enc_output, dec_output, enc_layer_att_scores, dec_layer_att_scores, dec_layer_cross_att_scores





from torch.utils.data import random_split
from torchvision import transforms

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
import numpy as np 

def load_data(fn):
    data = []
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            seq_str, sorted_seq_str = line.split('\t')
            seqs = seq_str.split(',')
            sorted_seqs = sorted_seq_str.split(',')
            data.append( (seqs, sorted_seqs) )
    return data


class NumberDataset(Dataset):
    """Dataset."""
    def __init__(self, fn, input_vocab, output_vocab, max_enc_seq_length, max_dec_seq_length):
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.max_enc_seq_length = max_enc_seq_length 
        self.max_dec_seq_length = max_dec_seq_length 
        
        # load 
        self.data = load_data(fn)

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx): 
        seq, sorted_seq = self.data[idx]
        sorted_seq_ids = [ self.output_vocab[t] for t in sorted_seq ]

        # [ input ] ---------------------------------------------
        # 
        # < encoder input > 
        seq_ids = [ self.input_vocab[t] for t in seq ]
        # <pad> processing
        pad_id      = self.input_vocab['<pad>']
        num_to_fill = self.max_enc_seq_length - len(seq)
        seq_ids     = seq_ids + [pad_id]*num_to_fill

        # mask processing (1 for valid, 0 for invalid)
        enc_pad_mask = [1]*len(seq) + [0]*num_to_fill

        # < decoder input> 
        #   - <start> should be added at first position
        dec_input_seq_ids = [ self.output_vocab['<start>']] + sorted_seq_ids 

        # <pad> processing
        pad_id = self.output_vocab['<pad>']
        num_to_fill = self.max_dec_seq_length - len(dec_input_seq_ids)
        dec_input_seq_ids = dec_input_seq_ids + [pad_id]*num_to_fill

        # [ output ] ---------------------------------------------
        #   - <end> should be  added at the last position
        pad_id = self.output_vocab['<pad>']

        dec_output_seq_ids = sorted_seq_ids + [ self.output_vocab['<end>']]
        
        # mask processing (1 for valid, 0 for invalid)
        # this mask can be used penalize unnecessary loss value at calcuation cross-entropy 
        dec_output_pad_mask = [1]*len(dec_output_seq_ids) + [0]*num_to_fill

        num_to_fill = self.max_dec_seq_length - len(dec_output_seq_ids)
        dec_output_seq_ids = dec_output_seq_ids + [pad_id]*num_to_fill

        item = [
                    # input - encoder 
                    np.array(seq_ids),
                    np.array(enc_pad_mask),  # encoder padding

                    # input - decoder
                    np.array(dec_input_seq_ids),

                    # output - decoder
                    np.array(dec_output_seq_ids),
                    np.array(dec_output_pad_mask),  # decoder output padding
               ]
        return item 


class NumberDataModule(pl.LightningDataModule):
    def __init__(self, 
                 max_enc_seq_length: int=20,
                 max_dec_seq_length: int=13, # for testing
                 batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.max_enc_seq_length = max_enc_seq_length 
        self.max_dec_seq_length = max_dec_seq_length 

        input_vocab, output_vocab = self.make_vocab('./data/sorted_numbers/train.txt')
        self.input_vocab_size = len( input_vocab )
        self.output_vocab_size = len( output_vocab )

        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

        self.enc_padding_idx = input_vocab['<pad>']
        self.dec_padding_idx = output_vocab['<pad>']

        self.all_train_dataset = NumberDataset('./data/sorted_numbers/train.txt', input_vocab, output_vocab, max_enc_seq_length, max_dec_seq_length)
        self.test_dataset      = NumberDataset('./data/sorted_numbers/test.txt', input_vocab, output_vocab, max_enc_seq_length, max_dec_seq_length)

        self.max_enc_seq_length = max_enc_seq_length
        self.max_dec_seq_length = max_dec_seq_length 


        # random split train / valiid for early stopping
        N = len(self.all_train_dataset)
        tr = int(N*0.8) # 8 for the training
        va = N - tr     # 2 for the validation 
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.all_train_dataset, [tr, va])

    def make_vocab(self, fn):
        input_tokens = []
        output_tokens = []
        data = load_data(fn)

        for seqs, sorted_seqs in data:
            for token in seqs:
                input_tokens.append(token)

            for token in sorted_seqs:
                output_tokens.append(token)
        
        input_tokens = list(set(input_tokens))
        output_tokens = list(set(output_tokens)) 

        input_tokens.sort()
        output_tokens.sort()

        # [encoder vocab]
        #   - add <pad> symbol to input tokens as a first item
        input_tokens = ['<pad>'] + input_tokens 
        input_vocab = { str(token):index for index, token in enumerate(input_tokens) }

        # [decoder vocab]
        output_tokens = ['<pad>', '<start>', '<end>'] + output_tokens  ## note that we need <start> and <end> for decoding
        output_vocab = { str(token):index for index, token in enumerate(output_tokens) }

        return input_vocab, output_vocab

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True) # NOTE : Shuffle

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

from torchmetrics import functional as FM


class Transformer_Number_Sorting(pl.LightningModule): 
    def __init__(self, 
                 input_vocab_size,
                 output_vocab_size,
                 num_layers,   # number of layers
                 d_model,      # dim. in attemtion mechanism 
                 num_heads, 
                 enc_padding_idx,
                 dec_padding_idx,
                 enc_max_length,
                 dec_max_length,

                 # optiimzer setting
                 learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()  

        ## embeddings for encoder and decoder (not shared so far)
        self.enc_emb = InputEmbeddings(
                                           vocab_size=self.hparams.input_vocab_size,
                                           hidden_size=self.hparams.d_model,
                                           pad_token_id=self.hparams.enc_padding_idx,
                                           max_length_size=self.hparams.enc_max_length,
                                           layer_norm_eps=1e-12,
                                           hidden_dropout_prob=0.1
                                       )
        self.dec_emb = InputEmbeddings(
                                           vocab_size=self.hparams.output_vocab_size,
                                           hidden_size=self.hparams.d_model,
                                           pad_token_id=self.hparams.dec_padding_idx,
                                           max_length_size=self.hparams.dec_max_length,
                                           layer_norm_eps=1e-12,
                                           hidden_dropout_prob=0.1
                                       )

        ## Transformer Block
        self.transformer = Transformer(
                                        num_layers=self.hparams.num_layers, 
                                        d_model=self.hparams.d_model, 
                                        num_heads=self.hparams.num_heads,
                                        dropout=0.1,
                                        dim_feedforward= 4*self.hparams.d_model
                                       )

        ## to output class
        self.to_output = nn.Linear(self.hparams.d_model, self.hparams.output_vocab_size) # D -> a single number

        # loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=dec_padding_idx)
        
    def get_class_weights(self, end_idx):
        ## weighting for <end> symbol output
        end_idx = 2 
        N = self.hparams.output_vocab_size 
        weight_for_end = 0.3
        weight_for_others = (1.0 - weight_for_end) / (N-1) # except end
        cls_weights = torch.ones(N) * weight_for_others
        cls_weights[end_idx] = weight_for_end
        return cls_weights

    def forward(self, enc_input_id, dec_input_id, enc_input_pad_mask):
        # ----------------------- ENCODING -------------------------------#
        # [ Digit Character Embedding ]
        #   for encoder and decoder
        enc_input = self.enc_emb(enc_input_id.long())
        dec_input = self.dec_emb(dec_input_id.long())
        
        enc_output, dec_output, layer_enc_attn_scores, layer_dec_attn_scores_1, layer_dec_attn_scores_2  =\
           self.transformer(enc_input, dec_input, enc_input_pad_mask)

        # to symbol 
        step_logits = self.to_output(dec_output) # [B, tar_seq_len, num_output_vocab]

        additional_outputs = [layer_enc_attn_scores, layer_dec_attn_scores_1, layer_dec_attn_scores_2]
        return step_logits, additional_outputs

    def training_step(self, batch, batch_idx):
        enc_input_ids,  enc_input_pad_mask, \
        dec_input_ids,  \
        dec_output_ids, dec_output_pad_mask = batch 

        step_logits, _ = self(enc_input_ids, dec_input_ids, enc_input_pad_mask)
        C = step_logits.size()[-1]
        loss = self.criterion(step_logits.view(-1, C), dec_output_ids.view(-1).long()) 

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        enc_input_ids,  enc_input_pad_mask, \
        dec_input_ids,  \
        dec_output_ids, dec_output_pad_mask = batch 

        step_logits, _ = self(enc_input_ids, dec_input_ids, enc_input_pad_mask)
        C = step_logits.size()[-1]
        loss = self.criterion(step_logits.view(-1, C), dec_output_ids.view(-1).long()) 
        
        ## get predicted result
        metrics = {'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def validation_step_end(self, val_step_outputs):
        val_loss = val_step_outputs['val_loss'].cpu()
        self.log('validation_loss', val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        enc_input_ids,  enc_input_pad_mask, \
        dec_input_ids,  \
        dec_output_ids, dec_output_pad_mask = batch 

        step_logits, _ = self(enc_input_ids, dec_input_ids, enc_input_pad_mask)
        C = step_logits.size()[-1]
        loss = self.criterion(step_logits.view(-1, C), dec_output_ids.view(-1).long()) 
        
        step_probs    = torch.softmax(step_logits, axis=-1) # [B, tar_seq_len, num_output_vocab]
        step_best_ids = torch.argmax(step_probs, axis=-1)

        ## prediction 
        result = {}
        result['input'] = enc_input_ids.cpu()
        result['predicted'] = step_best_ids.cpu()
        result['reference'] = dec_output_ids.cpu()
        result['step_probs'] = step_probs.cpu()
        return result

    def set_vocabs(self, input_vocab, output_vocab):
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

        self.r_input_vocab = {v:k for k, v in input_vocab.items() }
        self.r_output_vocab = {v:k for k, v in output_vocab.items() }


    def test_epoch_end(self, outputs):
        input      = result_collapse(outputs, 'input').cpu()
        predicted  = result_collapse(outputs, 'predicted').cpu()
        reference  = result_collapse(outputs, 'reference').cpu()
        step_probs = result_collapse(outputs, 'step_probs').cpu()

        def get_valid_items(tensor, pad_idx):
            a = tensor.data.cpu().numpy()
            a = a[ a != pad_idx ]
            return a 

        import os
        os.makedirs("./outputs", exist_ok=True)
        with open('./outputs/sorted_result.txt', 'w') as f:
            for _input, _pred, _ref, _prob in zip(input, predicted, reference, step_probs):
                _input = get_valid_items(_input, self.hparams.enc_padding_idx)
                _pred = get_valid_items(_pred, self.hparams.dec_padding_idx)
                _ref = get_valid_items(_ref, self.hparams.dec_padding_idx)

                ## trim _pred with first <end>
                _N = -1
                for idx, _i in enumerate(_pred):
                    if _i == self.output_vocab['<end>']: 
                        _N = idx
                        break
                _pred = _pred[:_N]

                input_seq = [ self.r_input_vocab[x] for x in _input ]
                pred_seq =  [ self.r_output_vocab[x] for x in _pred ]
                ref_seq =   [ self.r_output_vocab[x] for x in _ref if self.output_vocab['<end>'] != x ]

                
                input_seq = ",".join(input_seq)
                pred_seq = ",".join(pred_seq)
                ref_seq = ",".join(ref_seq)

                flag = 'O' if pred_seq == ref_seq else 'X'
                print(f'[{flag}] {input_seq}', file=f)
                print(f'\t\tREF  : {ref_seq}', file=f)
                print(f'\t\tPRED : {pred_seq}', file=f)
                print(f'-------------------', file=f)




    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Transformer")
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parent_parser



from argparse import ArgumentParser
from pytorch_lightning.callbacks import EarlyStopping
def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    ## Transformer is very sensitive to sorting task  
    ## Good settings so far 
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--d_model',    default=512, type=int)  # dim. for attention model -- 'H' at paper
    parser.add_argument('--num_heads',  default=8,   type=int)  # number of heads  -- 'A' at paper
    parser.add_argument('--num_layers', default=8,   type=int)  # number of layers -- 'L' at paper

    parser = pl.Trainer.add_argparse_args(parser)
    parser = Transformer_Number_Sorting.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = NumberDataModule.from_argparse_args(args)
    x = iter(dm.train_dataloader()).next() # <for testing 

    ## ------------
    ## model
    ## ------------
    model = Transformer_Number_Sorting(
                                    dm.input_vocab_size,
                                    dm.output_vocab_size,
                                    args.num_layers,    # number of layers 
                                    args.d_model,       # dim. in attemtion mechanism 
                                    args.num_heads,     # number of heads
                                    dm.enc_padding_idx,
                                    dm.dec_padding_idx,
                                    dm.max_enc_seq_length,
                                    dm.max_dec_seq_length,
                                    args.learning_rate
                                    )

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
                            max_epochs=1, 
                            callbacks=[EarlyStopping(monitor='val_loss')],
                            gpus = 1 # if you have gpu -- set number, otherwise zero
                        )
    trainer.fit(model, datamodule=dm)

    # copy 
    import shutil
    best_model_fn = trainer.checkpoint_callback.best_model_path
    import os; os.makedirs('./outputs/release/', exist_ok=True)
    shutil.copy(best_model_fn, './outputs/release/release.ckpt')

    # ------------
    # testing
    # ------------
    transformer_model = model.load_from_checkpoint('./outputs/release/release.ckpt')
    model.set_vocabs(dm.input_vocab, dm.output_vocab)
    result = trainer.test(model, test_dataloaders=dm.test_dataloader())



if __name__ == '__main__':
    cli_main()