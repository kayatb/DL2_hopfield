import argparse
import os
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import *
from models.transformers import HopfieldTransformerEncoder
import torch
import torch.nn as nn
import numpy as np


class TransformerModel(pl.LightningModule):
    def __init__(self, vocabulary, model, lr, lr_decay):
        super().__init__()
        self.save_hyperparameters()

        self.glove_embeddings = nn.Embedding.from_pretrained(vocabulary.vectors)

        if model == 'Hopfield':
            # Hardcoding the embedding dim
            self.encoder = HopfieldTransformerEncoder(d_model=300)
            print("Encoder Initialized")
            self.classifier = nn.Linear(300, 3)

        # classification loss function
        self.loss_function = nn.CrossEntropyLoss()

        # create instance to save the last validation accuracy. Needed for the PLCallback
        self.last_val_acc = None

    def forward(self, src, src_mask=None, src_padding_mask=None):
        # pass premises and hypothesis and get embedding vector
        x = self.embedding(src.text)
        x = self.encoder(x, src_mask, src_padding_mask)

        if self.reduction == "mean":
            x = x.mean(dim=0, keepdim=True).squeeze(0)
        else:
            raise ValueError(f"Invalid reduction method.")

        predictions = self.fc(x)
        loss = self.loss_function(predictions, src.label)
        label_predictions = torch.argmax(predictions, dim=1)
        # Get accuracy for the entire batch
        accuracy = torch.true_divide(torch.sum(label_predictions == src.label),
                                     torch.tensor(src.label.shape[0], device=src.label.device))
        return loss, accuracy

    # function that configures the optimizer for the model
    def configure_optimizers(self):
        # create optimizer
        optimizer = torch.optim.SGD([{'params': self.encoder.parameters()},
                                     {'params': self.classifier.parameters()}], lr=self.hparams.lr)

        # Don't update the embeddings weight only on the model
        self.glove_embeddings.weight.requires_grad = False

        # create learning rate decay
        lr_scheduler = {
            'scheduler': StepLR(optimizer=optimizer, step_size=1, gamma=self.hparams.lr_decay),
            'name': 'learning_rate'
        }

        # return the scheduler and optimizer
        return [optimizer], [lr_scheduler]

    # function that performs a training step
    def training_step(self, batch, batch_idx):
        # forward the batch through the model
        train_loss, train_acc = self.forward(batch)

        # log the training loss and accuracy
        self.log("train_loss", train_loss, on_step=False, on_epoch=True)
        self.log("train_acc", train_acc, on_step=False, on_epoch=True)

        # return the training loss
        return train_loss

    # function that performs a validation step
    def validation_step(self, batch, batch_idx):
        # forward the batch through the model
        val_loss, val_acc = self.forward(batch)

        # log the validation loss and accuracy
        self.log("val_loss", val_loss)
        self.log("val_acc", val_acc)

        # save the validation accuracy
        self.last_val_acc = val_acc

    # function that performs a test step
    def test_step(self, batch, batch_idx):
        # forward the batch through the model
        test_loss, test_acc = self.forward(batch)

        # log the test loss and accuracy
        self.log("test_loss", test_loss)
        self.log("test_acc", test_acc)


# TODO: Modify later
class PLCallback(pl.Callback):

    def __init__(self, lr_decrease_factor=5):
        """
        Inputs:
            lr_decrease_factor - Factor to divide the learning rate by when
                the validation accuracy decreases. Default is 5
        """
        super().__init__()

        # save the decrease factor
        self.decrease_factor = lr_decrease_factor

        # initialize the previous validation accuracy as 0
        self.last_val_acc = 0

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        """
        This function is called after every training epoch
        """

        # check if the learning rate has fallen under 10e-5
        current_lr = trainer.optimizers[0].state_dict()['param_groups'][0]['lr']
        if current_lr < 10e-5:
            # stop training
            trainer.should_stop = True

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        This function is called after every validation epoch
        """

        # If the validation accuracy is below the previous accuracy then continue
        if pl_module.last_val_acc < self.last_val_acc:
            # divide the learning rate by the specified factor
            state_dict = trainer.optimizers[0].state_dict()
            state_dict['param_groups'][0]['lr'] = state_dict['param_groups'][0]['lr'] / self.decrease_factor
            new_optimizer = torch.optim.SGD([{'params': pl_module.encoder.parameters()},
                                             {'params': pl_module.classifier.parameters()}],
                                            lr=state_dict['param_groups'][0]['lr'])
            new_optimizer.load_state_dict(state_dict)

            # update scheduler
            scheduler_state_dict = trainer.lr_schedulers[0]['scheduler'].state_dict()
            new_step_scheduler = StepLR(optimizer=new_optimizer, step_size=1, gamma=scheduler_state_dict['gamma'])
            new_step_scheduler.load_state_dict(scheduler_state_dict)
            new_scheduler = {
                'scheduler': new_step_scheduler,
                'name': 'learning_rate'
            }
            trainer.optimizers = [new_optimizer]
            trainer.lr_schedulers = trainer.configure_schedulers([new_scheduler])

        # save the validation accuracy
        self.last_val_acc = pl_module.last_val_acc


def train_model(args):
    print("STARTING")
    os.makedirs(args.log_dir, exist_ok=True)
    vocab, labels, train_iter, dev_iter, test_iter = load_snli(device=None, batch_size=args.batch_size)
    print("DATASET LOADED")
    # Checking the train_iter
    if not args.checkpoint_dir:
        pl_callback = PLCallback(lr_decrease_factor=args.lr_decrease_factor)
        monitor_lr = LearningRateMonitor(logging_interval='epoch')
        # Instantiate Trainer
        trainer = pl.Trainer(default_root_dir=args.log_dir,
                             checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                             gpus=1 if torch.cuda.is_available() else 0,
                             callbacks=[monitor_lr, pl_callback],
                             max_epochs=10,
                             progress_bar_refresh_rate=1 if args.progress_bar else 0)
        trainer.logger._default_hp_metric = None
        # create model
        pl.seed_everything(args.seed)
        model = TransformerModel(vocabulary=vocab, model=args.model, lr=args.lr, lr_decay=args.lr_decay)

        # train the model
        trainer.fit(model, train_iter, dev_iter)
        # load the best checkpoint
        model = TransformerModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    else:
        # create a PyTorch Lightning trainer
        trainer = pl.Trainer(logger=False,
                             checkpoint_callback=False,
                             gpus=1 if torch.cuda.is_available() else 0,
                             progress_bar_refresh_rate=1 if args.progress_bar else 0)

        # load model from the given checkpoint
        model = TransformerModel.load_from_checkpoint(args.checkpoint_dir)

    # test the model
    model.freeze()
    test_result = trainer.test(model, test_dataloaders=test_iter, verbose=True)

    # return the test results
    return test_result


# command line arguments parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', default='Hopfield', type=str,
                        help='What model to use. Default is AWE')

    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use. Default is 0.1')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Minibatch size. Default is 16')
    parser.add_argument('--lr_decay', default=0.99, type=float,
                        help='Learning rate decay after each epoch. Default is 0.99')
    # Decrease if the dev accuracy drops below lr_threshold
    parser.add_argument('--lr_decrease_factor', default=5, type=int,
                        help='Factor to divide learning rate by when dev accuracy decreases. Default is 5')
    # Parameter to hold the threshold which has to end the training
    parser.add_argument('--lr_threshold', default=1e-3, type=float,
                        help='Learning rate threshold after which model training is stopped. Default is 1e-5')

    parser.add_argument('--checkpoint_dir', default=None, type=str,
                        help='Directory where the model checkpoint is located. Default is None (no checkpoint used)')

    parser.add_argument('--seed', default=2022, type=int,
                        help='Seed to use for reproducing results. Default is 1234')
    parser.add_argument('--log_dir', default='pl_logs', type=str,
                        help='Directory where the PyTorch Lightning logs should be created. Default is pl_logs')
    parser.add_argument('--progress_bar', action='store_true',
                        help='Use a progress bar indicator for interactive experimentation.')
    args = parser.parse_args()

    train_model(args)
