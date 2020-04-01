# Python libraries
from statistics import mean

# External modules
import torch
from torch import nn
import torch.nn.functional as func
import torch.optim as optim

import pandas as pd
from tqdm import tqdm

# Internal modules
from util.PerformanceTracker import PerformanceTracker

# Typing
from typing import Tuple




import matplotlib.pyplot as plt




class ConvolutionalNeuralNet(nn.Module):

    def __init__(
        self,
        num_filters: Tuple[int, int, int, int],
        num_neurons_in_layers: Tuple[int, int],
        num_y_classes,
        device: torch.device
    ):
        num_1st_filter, num_2nd_filter, num_3rd_filter, num_4th_filter = num_filters
        num_1st_layer_neurons, num_2nd_layer_neurons = num_neurons_in_layers
        # num_1st_layer_neurons, num_2nd_layer_neurons, num_3rd_layer_neurons, num_4th_layer = num_neurons_in_layers

        super().__init__()
        self.conv1 = nn.Conv2d(3, num_1st_filter, 3, stride = 1)
        self.conv2 = nn.Conv2d(num_1st_filter, num_2nd_filter, 3)
        self.conv3 = nn.Conv2d(num_2nd_filter, num_3rd_filter, 3)
        self.conv4 = nn.Conv2d(num_3rd_filter, num_4th_filter, 3)

        # First dense input = [batch_size, height * width * num_channels]
        x = torch.randn(3, 128, 128).view(-1, 3, 128, 128)
        self._conv_out_len = None
        self._conv_forward(x)

        self.dense1 = nn.Linear(self._conv_out_len, num_1st_layer_neurons) 
        self.dense2 = nn.Linear(num_1st_layer_neurons, num_2nd_layer_neurons) 
        self.dense3 = nn.Linear(num_2nd_layer_neurons, num_y_classes)


    def _conv_forward(self, x: torch.tensor) -> torch.tensor:
        x = func.max_pool2d(func.relu(self.conv1(x)), (2, 2)) 
        x = func.max_pool2d(func.relu(self.conv2(x)), (2, 2)) 
        x = func.max_pool2d(func.relu(self.conv3(x)), (2, 2))
        x = func.max_pool2d(func.relu(self.conv4(x)), (2, 2))

        if self._conv_out_len is None:
            num_features = x[0].shape[0]
            num_px_height = x[0].shape[1]
            num_px_width = x[0].shape[2]
            self._conv_out_len = num_features * num_px_height * num_px_width

        return x

    
    def _prepare_conv_to_dense(self, x: torch.tensor) -> torch.tensor:
        return x.view(-1, self._conv_out_len)
    

    def _dense_forward(self, x: torch.tensor) -> torch.tensor:
        x = func.relu(self.dense1(x))
        x = func.relu(self.dense2(x))
        x = func.relu(self.dense3(x))

        return x


    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self._conv_forward(x)
        x = self._prepare_conv_to_dense(x)
        x = self._dense_forward(x)
        # x = self.output_activation(x)
        return x


    def train(
        self,
        train: Tuple[pd.DataFrame, pd.DataFrame], 
        val: Tuple[pd.DataFrame, pd.DataFrame],
        device: torch.device,
        tracker: PerformanceTracker,
        epochs: int,
        batch_size: int
    ) -> None:
        train_X, train_y = train
        train_X, train_y = torch.from_numpy(train_X), torch.from_numpy(train_y)

        val_X, val_y = val
        val_X, val_y = torch.from_numpy(val_X), torch.from_numpy(val_y)
        val_X, val_y = val_X.to(device), val_y.to(device)

        model = self
        model = model.to(device)
        model = model.float()
        
        optimizer = optim.Adam(model.parameters(), lr = 0.0001)
        loss_function = nn.CrossEntropyLoss() # multiclass, single label => categorical crossentropy as loss

        for epoch in range(epochs):
            batch_range = range(0, len(train_X), batch_size) # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
            train_losses = []
            train_accuracies = []

            for i in tqdm(batch_range):
                batch_X = train_X[i:i + batch_size].view(-1, 3, 128, 128).float()
                batch_y = train_y[i:i + batch_size].long()
                batch_y = torch.flatten(batch_y)

                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                # pred_y = func.softmax(pred_y, dim = 1)
                # Crossentropy berechnet intern schon den softmax 
                # https://discuss.pytorch.org/t/cross-entropy-loss-is-not-decreasing/43814/3
                model.zero_grad()
                pred_y = model(batch_X)
                
                loss = loss_function(pred_y, batch_y)
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step() # Does the update

                pred_y_indices = torch.argmax(pred_y, dim = 1)
                num_correct = int((pred_y_indices == batch_y).int().sum())
                # batch_size = int(batch_y.shape[0])
                accuracy = num_correct / batch_size
                train_accuracies.append(accuracy)

            val_loss, val_acc = model.validate((val_X, val_y), loss_function, device, batch_size)
            tracker.add_train(mean(train_losses), mean(train_accuracies))
            tracker.add_val(val_loss, val_acc)

            print(f'Epoch: {epoch + 1}')
            print(f'Train loss: {mean(train_losses)}')
            print(f'Train accuracy: {mean(train_accuracies)}')
            print(f'Val loss: {val_loss}')
            print(f'Val accuracy: {val_acc}')


    def validate(
        self, 
        val: Tuple[pd.DataFrame, pd.DataFrame], 
        loss_function, 
        device: torch.device,
        batch_size: int
    ) -> Tuple[float, float]:

        model = self
        val_X, val_y = val
        val_X, val_y = val_X.float(), val_y.long()
        val_X, val_y = val_X.to(device), val_y.to(device)

        correct = 0
        total = 0
        losses = []
        accuracies = []

        batch_range = range(0, len(val_X), batch_size)
        with torch.no_grad():
            for i in batch_range:
                batch_X = val_X[i:i + batch_size].view(-1, 3, 128, 128).float()
                batch_y = val_y[i:i + batch_size].long()
                batch_y = torch.flatten(batch_y)

                pred_y = model(batch_X)
                loss = loss_function(pred_y, batch_y)
                losses.append(loss.item())

                pred_y_indices = torch.argmax(pred_y, dim = 1)
                num_correct = int((pred_y_indices == batch_y).int().sum())
                accuracy = num_correct / batch_size
                accuracies.append(accuracy)

        return mean(losses), mean(accuracies)

