"""
    NLP 101 > N2M Translation (Neural Machine Translation) - Trainer

        - this code is for educational purpose.
        - the code is written for easy understanding not for optimized code.

    Author : Sangkeun Jung (hugmanskj@gmai.com)
    All rights reserved. (2022)
"""


# In this code, we will implement
#   - Sequence to Sequence learning based NMT
#   - Encoder will be based on BERT-like encoder (param. weights from pretrained model)
#   - Decoder will be based on Transformer-Original paper's decoder (param. weigths from the scracth!!!)
#
#   - All the symbols are already converted to ids 
#   - Check how to design input-output of the model 

from types import prepare_class
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# you can define any type of dataset
# dataset : return an example for batch construction. 
class NMTDataset(Dataset):
    """Dataset."""
    def __init__(self, data, enc_max_seq_len, dec_max_seq_len, input_pad_id, decoder_bos_id, decoder_eos_id):
        self.df = data 
        self.enc_max_seq_len = enc_max_seq_len
        self.dec_max_seq_len = dec_max_seq_len

        self.input_pad_id   = input_pad_id
        self.decoder_bos_id = decoder_bos_id
        self.decoder_eos_id = decoder_eos_id
        
        def add_pad(x, pad_id, max_len):
            T = len(x)
            N = max_len - T
            x += [pad_id]*N
            return x 

        def make_mask(x, max_len):
            T = len(x)
            N = max_len - T
            mask = [1] * T + [0]*N
            return mask 

        def add_bos_eos(x, bos_id, eos_id):
            x = [bos_id] + x + [eos_id]
            return x 

        def add_eos(x, eos_id):
            x = x + [eos_id]
            return x 

        ## input-output design
        ## example) A,B,C -> 1, 2, 3, 4, 5
        ## [input]
        ##  : A, B, C, #, #      <-- # is for padding symbol (in case of max length is 5) 
        ## [output]
        ##  <decoder input>
        ##  : <s>,  1, 2, 3, 4, 5, </s>, #, # <-- # is for padding symbol ( in case of max length 9)
        ##  : </s>, 1, 2, 3, 4, 5, </s>, #, # <-- in GPT2 tokenization case
        ##  <decoder output>
        ###    1,   2, 3, 4, 5, </s>, #, #, # <-- # is for padding symbol ( in case of max length 9)

        ## source processing (for ENCODER) ---
        self.source_masks     = self.df.source_input_ids.apply(lambda x : make_mask(x, enc_max_seq_len))
        self.source_token_ids = self.df.source_input_ids.apply(lambda x : add_pad(x, self.input_pad_id, enc_max_seq_len))
        # ignore token type ids since if it is none, it will handle as default in the forward process

        ## target processing (for DECODER) ---
        # <decoder input> 
        dec_pad_token_id = decoder_eos_id  # !! note that the dummy (padding) part will be handled with attention mask (so just set eos for now)
        bos_eos_target_token_ids   = self.df.target_input_ids.apply(lambda x : add_bos_eos(x, decoder_bos_id, decoder_eos_id))
        self.target_masks          = bos_eos_target_token_ids.apply(lambda x : make_mask(x, dec_max_seq_len))
        self.target_token_ids      = bos_eos_target_token_ids.apply(lambda x : add_pad(x, dec_pad_token_id, dec_max_seq_len))

        # <decoder output>
        dec_output_pad_id = -100 # cross entropy loss's default ignore label id 
        self.eos_target_token_ids = self.df.target_input_ids.apply(lambda x :  add_eos(x, decoder_eos_id))
        self.target_output_ids    = self.eos_target_token_ids.apply(lambda x : add_pad(x, dec_output_pad_id, dec_max_seq_len))

    def __len__(self):
        return len(self.target_output_ids) 

    def __getitem__(self, idx): 
        item = [
                    # input for encoder
                    torch.tensor(self.source_token_ids[idx]),
                    torch.tensor(self.source_masks[idx]),

                    # input for decoder
                    torch.tensor(self.target_token_ids[idx]),

                    # output for decoder 
                    torch.tensor(self.target_output_ids[idx])

                ]
        return item


class NMT_DataModule(pl.LightningDataModule):
    def __init__(self, 
                 batch_size: int = 32, 
                 enc_max_seq_len: int=150,
                 dec_max_seq_len: int=300
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.enc_max_seq_len = enc_max_seq_len
        self.dec_max_seq_len = dec_max_seq_len

    def prepare_data(self):
        # called only on 1 GPU
        fns={
                'train':'./data/translation/train.processed.pkl',
                'valid':'./data/translation/valid.processed.pkl',
                'test':'./data/translation/test.processed.pkl',

                'source_vocab': './data/translation/source.vocab',
                'target_vocab': './data/translation/target.vocab',
                'vocab_info': './data/translation/vocab.info.json',
            }

        # load pickled files
        import pandas as pd
        self.train_data = pd.read_pickle(fns['train']) 
        self.valid_data = pd.read_pickle(fns['valid'])
        self.test_data  = pd.read_pickle(fns['test'])

        #self.train_data = self.train_data[:100]
        self.test_data  = self.test_data[:30]

        print("TRAIN :", len(self.train_data))
        print("VALID :", len(self.valid_data))
        print("TEST  :", len(self.test_data))

        print("----- Train Data Statistics -----")
        print("[SOURCE] --- ")
        print(self.train_data.source_input_ids.apply(lambda x : len(x)).describe())
        print("[TARGET] --- ")
        print(self.train_data.target_input_ids.apply(lambda x : len(x)).describe())

        self.source_vocab, self.target_vocab, self.vocab_info = self.load_vocab(fns['source_vocab'], fns['target_vocab'], fns['vocab_info'])
        self.vocabs = (self.source_vocab, self.target_vocab, self.vocab_info)

        self.input_pad_id  = self.vocab_info['source']['pad']
        self.num_class = len(self.target_vocab)
        self.output_pad_id = -100 # pytorch's cross-entropy default value 


    def load_vocab(self, src_fn, tar_fn, info_fn):
        # load label vocab
        source_vocab = {}
        with open(src_fn, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                label, idx = line.split('\t')
                source_vocab[label] = int(idx)

        # load token vocab
        target_vocab = {}
        with open(tar_fn, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                label, idx = line.split('\t')
                target_vocab[label] = int(idx)

        import json
        with open(info_fn, 'r', encoding='utf-8') as f:
            vocab_info = json.load(f)

        return source_vocab, target_vocab, vocab_info

    def setup(self):
        # called on every GPU
        input_pad_id, decoder_bos_id, decoder_eos_id = self.vocab_info['source']['pad'], self.vocab_info['target']['bos'], self.vocab_info['target']['eos']
        self.train_dataset = NMTDataset(self.train_data, self.enc_max_seq_len, self.dec_max_seq_len, input_pad_id, decoder_bos_id, decoder_eos_id)
        self.valid_dataset = NMTDataset(self.valid_data, self.enc_max_seq_len, self.dec_max_seq_len, input_pad_id, decoder_bos_id, decoder_eos_id)
        self.test_dataset  = NMTDataset(self.test_data,  self.enc_max_seq_len, self.dec_max_seq_len, input_pad_id, decoder_bos_id, decoder_eos_id)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True) # NOTE : Shuffle

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


import torchmetrics

def result_collapse(outputs, target):
    if len( [x[target] for x in outputs][0].shape ) == 0:
        target_value = torch.stack([x[target] for x in outputs])
    else:
        target_value = torch.cat([x[target] for x in outputs])
    return target_value

class NeuralMachineTranslation(pl.LightningModule): 
    def __init__(self, 
                 output_vocab_size, 
                 output_pad_id,
                 enc_max_seq_len, 
                 dec_max_seq_len, 
                 # optiimzer setting
                 learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()  

        SOURCE_PRETRAINED_MODEL_NAME = 'klue/bert-base'  # source : Korean
        TARGET_PRETRAINED_MODEL_NAME = 'gpt2'            # target : English

        ## We will use BERT and GPT2 as basemodels for encoder and decoder, respectively. 
        ## [ENCODER] -----------------------------------------------------------------
        from transformers import BertModel, BertConfig
        hg_config = BertConfig.from_pretrained(SOURCE_PRETRAINED_MODEL_NAME)
        hg_bert = BertModel.from_pretrained(SOURCE_PRETRAINED_MODEL_NAME)
    
        from commons.bert import BERT, BERT_CONFIG
        my_config = BERT_CONFIG(hg_config=hg_config)
        self.encoder = BERT(my_config)
        self.encoder.copy_weights_from_huggingface(hg_bert)

        ## [DECODER] -----------------------------------------------------------------
        from commons.base import TransformerDecoder      # POST-layer normalization 
        emb_dim = 768
        self.decoder = TransformerDecoder(num_layers=8,  ## note this original decoder implemenatio does not include embedding and output parts
                                          d_model=768, 
                                          num_heads=8, 
                                          dropout=0.1, 
                                          dim_feedforward=768*4)

        ## GPT2-like embeddings
        self.dec_wte = nn.Embedding(output_vocab_size, emb_dim)  # input vocab size -> emb_dim
        self.dec_wpe = nn.Embedding(dec_max_seq_len, emb_dim)    # each position -> emb_dim
        self.dec_emb_dropout = nn.Dropout(0.1)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("dec_position_ids", torch.arange(dec_max_seq_len).expand((1, -1)))

        ## output
        self.to_output = nn.Linear(emb_dim, output_vocab_size) # D -> a single number

        self.criterion = nn.CrossEntropyLoss() # default -100 for padding output symbol

    def set_vocabs(self, vocabs):
        self.vocabs = vocabs 

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

    def forward_encoder(self, src_token_ids, src_mask):
        _, enc_output, enc_layer_attention_scores = self.encoder(src_token_ids, src_mask)
        # enc_output = [B, enc_seq_len, D]
        return enc_output

    def forward_decoder(self, enc_output, src_mask, tar_token_ids):
        # >> dec-input embedding
        token_embeddings = self.dec_wte(tar_token_ids) # each index maps to a (learnable) vector
        seq_length = tar_token_ids.shape[1]
        position_ids = self.dec_position_ids[:, :seq_length]
        position_embeddings = self.dec_wpe(position_ids) # each position maps to a (learnable) vector
        x = self.dec_emb_dropout(token_embeddings + position_embeddings)

        # >> decoder transformer 
        dec_length = tar_token_ids.size()[1] 
        look_ahead_mask = self.create_look_ahead_mask(dec_length).to(tar_token_ids.device)
        enc_pad_mask    = self.create_padding_mask(src_mask)

        seq_hidden_states, layers_attn_scores_1, layers_attn_scores_2 = self.decoder(x, 
                                                                                     enc_output, 
                                                                                     look_ahead_mask=look_ahead_mask, 
                                                                                     enc_pad_mask=enc_pad_mask)
        # seq_hidden_states = [B, dec_seq_len, D]

        # TO CLASS
        logits = self.to_output(seq_hidden_states) # logits = step_of_logits
        return logits, seq_hidden_states, layers_attn_scores_1, layers_attn_scores_2

    def forward(self, src_token_ids, src_mask, tar_token_ids):
        # --------
        # encoder
        # --------
        enc_output = self.forward_encoder(src_token_ids, src_mask)

        # --------
        # decoder
        # --------
        logits, seq_hidden_states, layers_attn_scores_1, layers_attn_scores_2 = self.forward_decoder(enc_output, src_mask, tar_token_ids)
        return logits

    def cal_loss(self, logits, label):
        # cross-entropy handle the final dimension specially.
        # final dimension should be compatible between logits and predictions

        ## --- handling padding label parts
        num_labels = logits.shape[-1]

        logits = logits.view(-1, num_labels) # [B*seq_len, logit_dim]
        label  = label.view(-1) # [B * seq_len] flatten 

        loss = self.criterion(logits, label)  # -100 for decoder output -padding

        return loss

    def training_step(self, batch, batch_idx):
        src_token_ids, src_mask, tar_token_ids, tar_output_ids = batch 
        logits = self(src_token_ids, src_mask, tar_token_ids)

        ## loss calculate by flatting sequential data
        loss = self.cal_loss(logits, tar_output_ids)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # all logs are automatically stored for tensorboard
        return loss

    def validation_step(self, batch, batch_idx):
        src_token_ids, src_mask, tar_token_ids, tar_output_ids = batch 
        logits = self(src_token_ids, src_mask, tar_token_ids)
        loss = self.cal_loss(logits, tar_output_ids)
        self.log('val_loss', loss )

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("N2N")
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
    parser.add_argument('--batch_size', default=25, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = NeuralMachineTranslation.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = NMT_DataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup()
    x = iter(dm.train_dataloader()).next() # <for testing 

    # ------------
    # model
    # ------------
    model = NeuralMachineTranslation(
                                dm.num_class,
                                dm.output_pad_id,
                                dm.enc_max_seq_len,
                                dm.dec_max_seq_len,
                                args.learning_rate
                           )
    model.set_vocabs(dm.vocabs) # explicitly set (do not include vocabs as hyperparameters)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
                            num_sanity_val_steps=0,
                            max_epochs=20, 
                            callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')],
                            gpus = 1 # if you have gpu -- set number, otherwise zero
                        )
    trainer.fit(model, datamodule=dm)

    # ------------
    # Store best model for further processing
    # ------------
    import shutil 
    to_model_fn =  "./result/nmt/best_model.ckpt"
    shutil.copyfile(trainer.checkpoint_callback.best_model_path, to_model_fn)
    print("[DUMP] model is dumped at ", to_model_fn)


if __name__ == '__main__':
    cli_main()