# Python libraries
from os import path

# Internal modules
from util.datasets import FirstAugmentedDataset
from util.use_gpu import detect_gpu, get_device
from modelling.model_2020_04_08_2.CNNAugDataRegularized import CNNAugDataRegularized
from util.PerformanceTracker import PerformanceTracker
from util.create_submit import create_submit


def main_train() -> None:
    detect_gpu()
    device = get_device()
    save_dir = path.join('modelling', 'model_2020_04_08_1')

    model = CNNAugDataRegularized((64, 64, 128, 256), (1024, 256), 4, device)
    data = FirstAugmentedDataset()
    tracker = PerformanceTracker(save_dir)
    model.train(data, 60, 20, tracker, learning_rate = 0.0001)
    
    tracker.graphs()
    tracker.save('metrics.csv')


# def main_predict() -> None:
#     detect_gpu()
#     device = get_device()
#     save_dir = path.join('modelling', 'model_2020_04_08_1')

#     model = CNNAugDataRegularized((32, 64, 128, 256), (1024, 128), 4, device)
#     data = FirstAugmentedDataset()
#     tracker = PerformanceTracker(save_dir)
#     # Number if epochs determined by minimum loss of in training.
#     model.train(data, 13, 20, tracker, learning_rate = 0.0001)
    
#     test_X, imgs_ids = data.get_test()
#     pred_y = model.predict(test_X)
#     create_submit(pred_y, imgs_ids, path.join(save_dir, 'submission.csv'))


if __name__ == '__main__':
    main_train()
    # main_predict()
