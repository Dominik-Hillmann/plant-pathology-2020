# Python libraries
from os import path

# External modules
from torchsummary import summary

# Internal modules
from util.datasets import FirstAugmentedDataset
from util.use_gpu import detect_gpu, get_device
from modelling.model_2020_05_10_1.VGGStyleNet import VGGStyleNet 
from util.PerformanceTracker import PerformanceTracker
from util.create_submit import create_submit


def main() -> None:
    detect_gpu()
    device = get_device()
    save_dir = path.join('modelling', 'model_2020_05_10_1')

    model = VGGStyleNet(4, device)
    summary(model.cuda(), (3, 128, 128))
    print(model)
    data = FirstAugmentedDataset()
    tracker = PerformanceTracker(save_dir)
    model.train(data, 40, 64, tracker, learning_rate = 0.00001)
    
    tracker.graphs()
    tracker.save('metrics.csv')

    test_X, imgs_ids = data.get_test()
    pred_y = model.predict(test_X)
    create_submit(pred_y, imgs_ids, path.join(save_dir, 'submission.csv'))


if __name__ == '__main__':
    main()
