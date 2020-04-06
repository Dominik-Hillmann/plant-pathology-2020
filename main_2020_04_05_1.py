# Python libraries
from os import path

# Internal modules
from util.datasets import FirstAugmentedDataset
from util.use_gpu import detect_gpu, get_device
from modelling.model_2020_04_05_1.CNNAugData import CNNAugData
from util.PerformanceTracker import PerformanceTracker
from util.create_submit import create_submit


def main() -> None:
    detect_gpu()
    device = get_device()
    model = CNNAugData((64, 128, 512, 1024), (1024, 1024), 4, device)
    data = FirstAugmentedDataset()
    tracker = PerformanceTracker(path.join('modelling', 'model_2020_04_05_1'))
    model.train(data, 20, 20, tracker, learning_rate = 0.001)
    tracker.graphs()
    tracker.save('metrics.csv')


if __name__ == '__main__':
    main()
