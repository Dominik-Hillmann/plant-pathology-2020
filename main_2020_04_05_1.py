# Python libraries
from os import path

# Internal modules
from util.datasets import FirstAugmentedDataset
from util.use_gpu import detect_gpu, get_device
from modelling.model_2020_04_05_1.CNNAugData import CNNAugData
from util.PerformanceTracker import PerformanceTracker
from util.create_submit import create_submit


def main_train() -> None:
    detect_gpu()
    device = get_device()
    save_dir = path.join('modelling', 'model_2020_04_05_1')

    model = CNNAugData((16, 32, 64, 128), (512, 128), 4, device)
    data = FirstAugmentedDataset()
    tracker = PerformanceTracker(save_dir)
    model.train(data, 1, 20, tracker, learning_rate = 0.0001)

    tracker.graphs()
    tracker.save('metrics.csv')


def main_predict() -> None:
    detect_gpu()
    device = get_device()
    save_dir = path.join('modelling', 'model_2020_04_05_1')

    model = CNNAugData((16, 32, 64, 128), (512, 128), 4, device)
    data = FirstAugmentedDataset()
    tracker = PerformanceTracker(save_dir)
    # Number if epochs determined by minimum loss of in training.
    model.train(data, 1, 20, tracker, learning_rate = 0.0001)
    
    test_X, imgs_ids = data.get_test()
    pred_y = model.predict(test_X)
    create_submit(pred_y, imgs_ids, path.join(save_dir, 'submission.csv'))


if __name__ == '__main__':
    main_train()
    main_predict()
