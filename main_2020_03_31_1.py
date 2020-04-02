# Python libraries
import os

# Internal modules
from util.get_128px_data import get_128px_test_data, get_128px_train_data
from util.use_gpu import detect_gpu, get_device
from modelling.model_2020_03_31_1.ConvolutionalNeuralNet import ConvolutionalNeuralNet
from util.PerformanceTracker import PerformanceTracker
from util.create_submit import create_submit

detect_gpu()
device = get_device()
model = ConvolutionalNeuralNet(
    (64, 128, 512, 1024), 
    (1024, 1024),
    4,
    device
)

def main_try() -> None:
    train_X, train_y, val_X, val_y = get_128px_train_data()
    tracker = PerformanceTracker(os.path.join('modelling', 'model_2020_03_31_1'))
    model.train(
        (train_X, train_y),
        45, 
        10,
        val = (val_X, val_y), 
        tracker = tracker
    )

    tracker.graphs()
    tracker.save('metrics.csv')


def main_full() -> None:
    train_X, train_y, val_X, val_y = get_128px_train_data(val_size = 2)
    model.train((train_X, train_y), 1, 10) 

    del train_X; del train_y; del val_X; del val_y

    test_X, test_y = get_128px_test_data()
    pred_y = model.predict(test_X)
    create_submit(pred_y, test_y)


if __name__ == '__main__':
    # main_try()
    main_full()
