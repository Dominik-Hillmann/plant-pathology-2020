# Python libraries
from os import path
from statistics import mean
import csv

# External modules
import pandas as pd
import matplotlib.pyplot as plt


class PerformanceTracker:
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.epoch_train_losses = []
        self.epoch_train_acc = []
        self.epoch_val_losses = []
        self.epoch_val_acc = []

    
    def add_train(self, loss, acc):
        self.epoch_train_losses.append(loss)
        self.epoch_train_acc.append(acc)

    
    def add_val(self, loss, acc):
        self.epoch_val_losses.append(loss)
        self.epoch_val_acc.append(acc)


    def save(self, file_name):
        save_frame = pd.DataFrame({
            'losses_train': self.epoch_train_losses,
            'accuracies_train': self.epoch_train_acc,
            'losses_val': self.epoch_val_losses,
            'accuracies_val': self.epoch_val_acc
        })

        save_frame.to_csv(path.join(self.save_dir, file_name), quoting = csv.QUOTE_ALL)


    def graphs(self):
        
        epochs = list(range(1, self._get_num_epochs() + 1))
        # Losses        
        plt.plot(epochs, self.epoch_train_losses, 'r-', label = 'Training loss')
        plt.plot(epochs, self.epoch_val_losses, 'b-', label = 'Validation loss')
        plt.title('Average cross entropy loss by epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss metric')
        plt.legend(loc = 'upper right')
        plt.savefig(path.join(self.save_dir, 'losses.png'))

        plt.clf()

        # Accuracies
        plt.plot(epochs, self.epoch_train_acc, 'r-', label = 'Training accuracy')
        plt.plot(epochs, self.epoch_val_acc, 'b-', label = 'Validation accuracy')
        plt.title('Average accuracy by epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc = 'lower right')
        plt.savefig(path.join(self.save_dir, 'accuracies.png'))

    
    def print_stats(self, epoch: int) -> None:
        last_train_loss = self.epoch_train_losses[len(self.epoch_train_losses) - 1]
        last_val_loss = self.epoch_val_losses[len(self.epoch_val_losses) - 1]
        last_train_acc = self.epoch_train_acc[len(self.epoch_train_acc) - 1]
        last_val_acc = self.epoch_val_acc[len(self.epoch_val_acc) - 1]
        # print(self.epoch_train_losses)
        # print(self.epoch_train_acc)
        # print(self.epoch_val_losses)
        # print(self.epoch_val_acc)
        print('{0:>21}: {1}'.format('Epoch', round(epoch + 1, 3)))
        print('{0:>21}: {1}'.format('Training loss', round(last_train_loss, 5)))
        print('{0:>21}: {1}'.format('Training accuracy', round(last_train_acc, 5)))
        print('{0:>21}: {1}'.format('Validation loss', round(last_val_loss, 5)))
        print('{0:>21}: {1}'.format('Validation accuracy', round(last_val_acc, 5)))
    

    def _get_num_epochs(self):
        num_train_loss = len(self.epoch_train_losses)
        num_train_acc = len(self.epoch_train_acc)
        num_val_losses = len(self.epoch_val_losses)
        num_val_acc = len(self.epoch_val_acc)
        same_lengths = (
            num_train_loss ==
            num_train_acc ==
            num_val_losses ==
            num_val_acc
        )

        if not same_lengths:
            raise ValueError('Not all metrics have the same number of observations.')
            
        return num_train_loss
