import mxnet as mx
from mxnet.gluon.data.vision.datasets import ImageFolderDataset
from .cifar100 import get_transform_test
from multiprocessing import cpu_count
import os

def get_test_image_loader(path, num_workers):
    test_transform = get_transform_test()
    dataset = ImageFolderDataset(path).transform_first(test_transform)
    data_loader = mx.gluon.data.DataLoader(dataset, batch_size=1, num_workers=num_workers)
    return data_loader
