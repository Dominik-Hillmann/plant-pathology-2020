# Python libraries
from statistics import mean
from copy import deepcopy

# External modules
import torch
from torch import nn
import torch.nn.functional as func
import torch.optim as optim
from torchsummary import summary

import pandas as pd
import numpy as np
from tqdm import tqdm

# Internal modules
from util.PerformanceTracker import PerformanceTracker
from util.datasets import Dataset

# Typing
from typing import Tuple


class VGGStyleNet(nn.Module):

    def __init__(
        self,
        num_y_classes: int,
        device: torch.cuda.device,
        loss_function = nn.CrossEntropyLoss(),
        last_layer_activation = nn.Softmax(dim = 1)
    ):
        super().__init__()
        self.loss_function = loss_function
        self.activation = last_layer_activation
        self.used_device = device

        # Layer group 1
        first_layer_channels = 64
        self.conv1a = nn.Conv2d(
            3, 
            first_layer_channels, 
            kernel_size = 3, 
            stride = 1
        )
        self.conv1b = nn.Conv2d(
            in_channels = first_layer_channels,
            out_channels = first_layer_channels,
            kernel_size = 3, 
            stride = 1
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        # Layer group 2
        second_layer_channels = 128
        self.conv2a = nn.Conv2d(
            in_channels = first_layer_channels, 
            out_channels = second_layer_channels, 
            kernel_size = 3, 
            stride = 1
        )
        self.conv2b = nn.Conv2d(
            in_channels = second_layer_channels,
            out_channels = second_layer_channels,
            kernel_size = 3, 
            stride = 1
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        # Layer group 3
        thrid_layer_channels = 256
        self.conv3a = nn.Conv2d(
            in_channels = second_layer_channels, 
            out_channels = thrid_layer_channels, 
            kernel_size = 3, 
            stride = 1
        )
        self.conv3b = nn.Conv2d(
            in_channels = thrid_layer_channels,
            out_channels = thrid_layer_channels,
            kernel_size = 3, 
            stride = 1
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        # Layer group 4
        fourth_layer_channels = 512
        self.conv4a = nn.Conv2d(
            in_channels = thrid_layer_channels, 
            out_channels = fourth_layer_channels, 
            kernel_size = 3, 
            stride = 1
        )
        self.conv4b = nn.Conv2d(
            in_channels = fourth_layer_channels,
            out_channels = fourth_layer_channels,
            kernel_size = 3, 
            stride = 1
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        # First dense input = [batch_size, height * width * num_channels]
        x = torch.randn(3, 128, 128).view(-1, 3, 128, 128)
        self._conv_out_len = None
        self._conv_forward(x)

        last_layer_neurons = 256
        self.dense1 = nn.Linear(self._conv_out_len, last_layer_neurons) 
        self.dense_out = nn.Linear(last_layer_neurons, num_y_classes)


    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = func.relu(self.conv1a(x))
        x = func.relu(self.conv1b(x))
        x = self.pool1(x)

        x = func.relu(self.conv2a(x))
        x = func.relu(self.conv2b(x))
        x = self.pool2(x)

        x = func.relu(self.conv3a(x))
        x = func.relu(self.conv3b(x))
        x = self.pool3(x)

        x = func.relu(self.conv4a(x))
        x = func.relu(self.conv4b(x))
        x = self.pool4(x)

        if self._conv_out_len is None:
            print('In:', x.shape)
            num_features = x[0].shape[0]
            num_px_height = x[0].shape[1]
            num_px_width = x[0].shape[2]
            self._conv_out_len = num_features * num_px_height * num_px_width
            print('Out:', num_features * num_px_height * num_px_width)

        return x

    
    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, self._conv_out_len)
    

    def _dense_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = func.relu(self.dense1(x))
        x = func.relu(self.dense_out(x))

        return x


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_forward(x)
        x = self._flatten(x)
        x = self._dense_forward(x)

        return x


    def train(
        self,
        data: Dataset, 
        epochs: int,
        batch_size: int,
        tracker: PerformanceTracker = None,
        learning_rate = 0.001
    ) -> None:
        val_X, val_y = data.get_val()
        val_X, val_y = torch.from_numpy(val_X), torch.from_numpy(val_y)
        val_X = val_X.permute(0, 3, 1, 2)
        val_X, val_y = val_X.float(), val_y.long()
        # Assigned to GPU in batches in validate method.

        model = self
        model = model.to(self.used_device)
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        num_batches = data.num_possible_batches(batch_size)
        batch_range = range(num_batches) # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev

        for epoch in range(epochs):
            train_losses = []
            train_accuracies = []
            epoch_data = deepcopy(data)

            for _ in tqdm(batch_range):
                batch_X, batch_y = epoch_data.get_next_batch(batch_size)
                batch_X, batch_y = torch.from_numpy(batch_X), torch.from_numpy(batch_y)
                batch_X = batch_X.permute(0, 3, 1, 2)
                batch_X, batch_y = batch_X.float(), batch_y.long()
                batch_X, batch_y = batch_X.to(self.used_device), batch_y.to(self.used_device)

                model.zero_grad()
                pred_y = model(batch_X)
                loss =  self.loss_function(pred_y, batch_y)
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
                pred_y_indices = torch.argmax(pred_y, dim = 1)

                num_correct = int((pred_y_indices == batch_y).int().sum())
                accuracy = num_correct / batch_size
                train_accuracies.append(accuracy)

            print('Batch prediction:')
            print(pred_y)
            val_loss, val_acc = model.validate(val_X, val_y, batch_size)
            tracker.add_train(mean(train_losses), mean(train_accuracies))
            tracker.add_val(val_loss, val_acc)
            tracker.print_stats(epoch)


    def validate(self, val_X: torch.Tensor, val_y: torch.Tensor, batch_size: int) -> Tuple[float, float]:
        model = self
        correct = 0
        total = 0
        losses = []
        accuracies = []

        batch_range = range(0, len(val_X), batch_size)
        with torch.no_grad():
            for i in batch_range:
                batch_X = val_X[i:i + batch_size].float()
                batch_y = val_y[i:i + batch_size].long()
                batch_X, batch_y = batch_X.to(self.used_device), batch_y.to(self.used_device)

                pred_y = model(batch_X)
                loss = self.loss_function(pred_y, batch_y)
                losses.append(loss.item())

                pred_y_indices = torch.argmax(pred_y, dim = 1)
                num_correct = int((pred_y_indices == batch_y).int().sum())
                accuracy = num_correct / batch_size
                accuracies.append(accuracy)

        return mean(losses), mean(accuracies)


        model = self
        with torch.no_grad():
            pred_y = model(val_X)
            loss = self.loss_function(pred_y, val_y)

        loss_value = loss.item()
        pred_y_indices = torch.argmax(pred_y, dim = 1)
        num_correct = int((pred_y_indices == batch_y).int().sum())
        accuracy = num_correct / batch_size

        return loss_value, accuracy
    

    def predict(self, test_X: np.array) -> torch.Tensor:
        model = self
        model.to(self.used_device)
        
        preds = []
        batch_range = range(len(test_X))

        with torch.no_grad():
            for i in batch_range:
                batch_X = test_X[i:(i + 1)] # Gets observation as in [i] but keeps dimensionality.
                batch_X = torch.from_numpy(batch_X)
                batch_X = batch_X.permute(0, 3, 1, 2)
                batch_X = batch_X.float()
                batch_X = batch_X.to(self.used_device)

                pred_y = model(batch_X)
                pred_y = self.activation(pred_y)
                preds.append(pred_y)
        
        for i, pred in enumerate(preds):
            preds[i] = pred.cpu().numpy()[0]

        preds = np.array(preds)
        return preds


def main():
    model = VGGStyleNet()
    summary(model, (3, 128, 128))

if __name__ == '__main__':
    main()
