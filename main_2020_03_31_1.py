# Python libraries
import os

# Internal modules
from util.get_128px_data import get_128px_test_data, get_128px_train_data
from util.use_gpu import detect_gpu, get_device
from modelling.model_2020_03_31_1.ConvolutionalNeuralNet import ConvolutionalNeuralNet
from util.PerformanceTracker import PerformanceTracker


def main() -> None:
    detect_gpu()
    device = get_device()

    model = ConvolutionalNeuralNet((64, 32, 16), (1024, 4), device)
    train_X, train_y, val_X, val_y = get_128px_train_data()
    tracker = PerformanceTracker(os.path.join('modelling', 'model_2020_03_31_1'))

    model.train(
        (train_X, train_y), 
        (val_X, val_y), 
        device, tracker, 
        2, 
        10
    )


if __name__ == '__main__':
    main()
