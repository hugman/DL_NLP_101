"""
    NLP 101 > N21 Classification (TOPIC Classfication) 

        - this code is for educational purpose.
        - the code is written for easy understanding not for optimized code.

    Author : Sangkeun Jung (hugmanskj@gmai.com)
    All rights reserved. (2021)
"""


# In this code, we will implement
#   - BERT-based classification
#   - We will re-use our BERT code
#   - We will import pre-trained BERT weights of huggingface to our BERT
#   - Check how to prepare data and process


from types import prepare_class
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

PRETRAINED_MODEL_NAME = 'klue/bert-base'

# you can define any type of dataset
# dataset : return an example for batch construction. 
class TopicDataset(Dataset):
    """Dataset."""
    def __init__(self, data, tokenizer, label_vocab, max_seq_length):
        texts  = [x[0] for x in data]
        labels = [x[1] for x in data]
        self.inputs = tokenizer(texts, padding='max_length', truncation=True, return_tensors='pt', max_length=max_seq_length)
        self.outputs = [ label_vocab.get(x) for x in labels ]

    def __len__(self):
        return len(self.outputs) 

    def __getitem__(self, idx): 
        item = [
                    # input
                    self.inputs.input_ids[idx],
                    self.inputs.token_type_ids[idx],
                    self.inputs.attention_mask[idx], 
             
                    # output
                    torch.tensor(self.outputs[idx])
                ]
        return item


class KLEU_TopicDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, max_seq_length : int=512):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

    def prepare_data(self):
        # called only on 1 GPU
        # load dataset to current CPU
        self.train_data = self.load_data('./data/classification/train.tsv')
        self.valid_data = self.load_data('./data/classification/valid.tsv')
        self.test_data  = self.load_data('./data/classification/test.tsv')

        # prepare tokenizer
        # checkout : https://github.com/kiyoungkim1/LMkor
        from transformers import BertTokenizerFast
        self.tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME)

        # prepare label dictionary
        labels = sorted( list(set([x[1] for x in self.train_data]) ) )
        self.label_vocab = { key:idx for idx, key in enumerate(labels)}
        self.num_class = len(self.label_vocab)

    def setup(self, stage = None):
        # called on every GPU
        self.train_dataset = TopicDataset(self.train_data, self.tokenizer, self.label_vocab, self.max_seq_length)
        self.valid_dataset = TopicDataset(self.valid_data, self.tokenizer, self.label_vocab, self.max_seq_length)
        self.test_dataset  = TopicDataset(self.test_data,  self.tokenizer, self.label_vocab, self.max_seq_length)

    def load_data(self, fn):
        data = []
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                fields = line.split('\t')

                ## due to current data error
                if len(fields) > 2 :
                    text, label = fields[0], fields[-1]
                else:
                    text, label = fields

                data.append( (text, label) )
        return data 

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True) # NOTE : Shuffle

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

from torchmetrics import functional as FM

from commons.bert import BERT, BERT_CONFIG
class TopicClassifier(pl.LightningModule): 
    def __init__(self, 
                 num_class, 
                 # optiimzer setting
                 learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()  

        ## [text encoder] 
        ## prepare pretrained - TRANSFORMER Model
        from transformers import BertModel, BertConfig
        hg_config = BertConfig.from_pretrained(PRETRAINED_MODEL_NAME)
        hg_bert   = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
    
        ## ours
        my_config    = BERT_CONFIG(hg_config=hg_config)
        self.encoder = BERT(my_config)
        self.encoder.copy_weights_from_huggingface(hg_bert)  ## !! important

        ## BERT has pooler network emplicitly. 
        ## if your transformer does not have pooler, you can use top-layer's specific output.
        pooled_dim = self.encoder.pooler.dense.weight.shape[-1]

        # [to output]
        self.to_output = nn.Linear(pooled_dim, self.hparams.num_class) # D -> a single number

        # loss
        self.criterion = nn.CrossEntropyLoss()  

    def forward(self, input_ids, token_type_ids, attention_mask):
        # ENCODING with Transformer 
        pooled_output, _, layers_attention_scores = self.encoder(
                                                    input_ids=input_ids, 
                                                    token_type_ids=token_type_ids,
                                                    attention_mask=attention_mask
                                                    )

        # TO CLASS
        logits = self.to_output(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, label = batch 
        logits = self(input_ids, token_type_ids, attention_mask)
        loss   = self.criterion(logits, label.long()) 

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # all logs are automatically stored for tensorboard
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, label = batch 
        logits = self(input_ids, token_type_ids, attention_mask)
        loss = self.criterion(logits, label.long()) 
        
        ## get predicted result
        prob = F.softmax(logits, dim=-1)
        acc  = FM.accuracy(prob, label)
        f1   = FM.f1(prob, label)
        metrics = {'val_acc': acc, 'val_f1':f1, 'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def validation_step_end(self, val_step_outputs):
        val_acc  = val_step_outputs['val_acc'].cpu()
        val_f1   = val_step_outputs['val_f1'].cpu()
        val_loss = val_step_outputs['val_loss'].cpu()

        self.log('validation_acc',  val_acc, prog_bar=True)
        self.log('validation_f1',  val_f1, prog_bar=True)
        self.log('validation_loss', val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, label = batch 
        logits = self(input_ids, token_type_ids, attention_mask)
        loss = self.criterion(logits, label.long()) 
        
        ## get predicted result
        prob = F.softmax(logits, dim=-1)
        acc = FM.accuracy(prob, label)
        f1  = FM.f1(prob, label)
        metrics = {'test_acc': acc, 'test_f1': f1, 'test_loss': loss}
        self.log_dict(metrics, on_epoch=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("N21")
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
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--max_seq_length', default=200, type=int)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = TopicClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = KLEU_TopicDataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup(stage='fit')
    x = iter(dm.train_dataloader()).next() # <for testing 

    # ------------
    # model
    # ------------
    model = TopicClassifier(
                                dm.num_class,
                                args.learning_rate
                           )

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


    ## checkout benchmark performance 
    ## https://github.com/KLUE-benchmark/KLUE
    ## In case of KLUE-BERT-base, officail benchmark performance is 85.73 F1 score
    ##
    ## ours : [{'test_acc': 0.8512133359909058, 'test_f1': 0.8512133359909058, 'test_loss': 0.5294674038887024}]
    ## but you can increase the performance easily by controllong learning rate, batch size and other parameters. 


if __name__ == '__main__':
    cli_main()