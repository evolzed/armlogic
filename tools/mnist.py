import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import cv2
import torchvision.transforms as transforms
import operator
batch_size = 100
from tools.KNN import *
MnistDataSetDir = 'E:/ml/pymnist'

train_dataset = dsets.MNIST(root=MnistDataSetDir,
                            train=True,
                            transform=None,
                            download=True)
test_dataset = dsets.MNIST(root=MnistDataSetDir,
                           train=False,
                           transform=None,
                           download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

def getMnistData():
    # mnist数据集转Numpy
    X_train = train_loader.dataset.train_data.numpy()

    # mnist数据集改形状
    x_train = X_train.reshape(X_train.shape[0], 28 * 28)

    # mnist标签集转numpy
    x_label = train_loader.dataset.train_labels.numpy()
    # 归一化
    x_train = meanNorm(x_train, x_train)

    return x_train, x_label
