'''
    Machine Learning Block Implementation Practice
    with Pytorch Lightning

    Author : Sangkeun Jung (2021)
'''

# most of the case, you just change the component loading part
# all other parts are almost same
#

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader

# you can define any type of dataset
# dataset : return an example for batch construction. 
class MNISTDataset(Dataset):
    """MNIST dataset.
        a single example : python object --> Tensor  
    """

    def __init__(self, fn_dict):
        # load 
        import numpy as np 
        self.image_data = np.load(fn_dict['image'])  
        self.label_data = np.load(fn_dict['label'])

    def __len__(self):
        return len(self.image_data) # <-- this is important!!

    def __getitem__(self, idx): # <-- !!!! important function.
        image = self.image_data[idx]
        label = self.label_data[idx]

        # normalize
        # 1-2) preprocessing (2D --> 1D)
        image = image.reshape(784)  
        image = image.astype('float32')

        # 1-2) preprocessing (normalize to 0~1.0)
        image /= 255

        sample = [image, label]
        return sample

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, 
                 batch_size: int = 32):
        super().__init__()

        # in case of MNIST, 
        # you don't need complext process.
        # you can just use MNIST dataset of pytorch, but in this tutorial
        # I will show very detail process for education purpose. 
        

        fns = {
                'train' : 
                { 
                    'image' : f'./mnist/data/train.image.npy',
                    'label' : f'./mnist/data/train.label.npy'
                },
                'test' : 
                {
                    'image' : f'./mnist/data/test.image.npy',
                    'label' : f'./mnist/data/test.label.npy'
                }
        }

        self.batch_size = batch_size

        ## NOTE 
        ## --- Pytorch lightning provides '*.prepare_data()' and '*.setup()'
        ## --- for advanced applications, you may need those methods. 
        ## but, in here, we just load all data at init step for readibility 
        ## check prepare_data(), setup() <-- pytorch lightning official document
       
        # numpy object to custom DATASET
        self.all_train_dataset = MNISTDataset(fns['train'])
        self.test_dataset      = MNISTDataset(fns['test'])

        # random split train / valiid for early stopping
        N = len(self.all_train_dataset)
        tr = int(N*0.8) # 8 for the training
        va = N - tr     # 2 for the validation 
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.all_train_dataset, [tr, va])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True) # NOTE : Shuffle

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def teardown(self):
        # Used to clean-up when the run is finished
        ...

from pytorch_lightning.metrics import functional as FM

# pl.LightningModule is inherited from the nn.Module

class MLP_MNIST_Classifier(pl.LightningModule): 
    # <-- note that nn.module --> pl.LightningModule
    def __init__(self, 

                 ## ----------> 
                 learning_rate=1e-3
                 ## <---------                 
                 ):
        super().__init__()
        self.save_hyperparameters()  # <-- it store arguments to self.hparams.* 

        # network design here 
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        # loss
        self.criterion = nn.CrossEntropyLoss()  

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        
        return x

    def training_step(self, batch, batch_idx):
        # NOTE : "training_step" is "RESERVED"
        # batch_idx is sometimes needed. 
        image, label = batch 
        label_logits = self(image)  # <-- self call self.forward !
        loss = self.criterion(label_logits, label.long()) 


        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # all logs are automatically stored for tensorboard
        return loss

    def validation_step(self, batch, batch_idx):
        # NOTE : "validation_step" is "RESERVED"
        image, label = batch   # image : [batch_size, 784], label=[batch_size]
        label_logits = self(image)  
        loss = self.criterion(label_logits, label.long()) 
        ## get predicted result
        prob = F.softmax(label_logits, dim=-1)
        acc = FM.accuracy(prob, label)

        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def validation_step_end(self, val_step_outputs):
        val_acc  = val_step_outputs['val_acc'].cpu()
        val_loss = val_step_outputs['val_loss'].cpu()

        self.log('validation_acc',  val_acc, prog_bar=True)
        self.log('validation_loss', val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # NOTE : "test_step" is "RESERVED"
        image, label = batch   # image : [batch_size, 784], label=[batch_size]
        label_logits = self(image)  
        loss = self.criterion(label_logits, label.long()) 
        prob = F.softmax(label_logits, dim=-1)
        acc = FM.accuracy(prob, label)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MLP_MNIST_Classifier")
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
    parser.add_argument('--a1', default=200, type=int)
    parser.add_argument('--a2', default=200, type=int)
    parser.add_argument('--a3', default=200, type=int)
    parser.add_argument('--a4', default=200, type=int)
    parser.add_argument('--a5', default=200, type=int)




    parser = pl.Trainer.add_argparse_args(parser)
    parser = MLP_MNIST_Classifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = MNISTDataModule.from_argparse_args(args)

    # ------------
    # model
    # ------------
    model = MLP_MNIST_Classifier(args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
                            max_epochs=50, 
                            callbacks=[EarlyStopping(monitor='val_loss')],
                            gpus = 1 # if you have gpu -- set number, otherwise zero
                        )
    trainer.fit(model, datamodule=dm)

    # ------------
    # testing
    # ------------
    result = trainer.test(model, test_dataloaders=dm.test_dataloader())
    print(result)


if __name__ == '__main__':
    cli_main()