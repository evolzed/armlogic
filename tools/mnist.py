import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import cv2
import torchvision.transforms as transforms
import operator
batch_size = 10
from tools.KNN import *
MnistDataSetDir = 'E:/ml/pymnist'
from torch.autograd import Variable

train_dataset = dsets.MNIST(root=MnistDataSetDir,
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
test_dataset = dsets.MNIST(root=MnistDataSetDir,
                           train=False,
                           transform=transforms.ToTensor(),
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
    print(X_train.shape[0])

    # mnist数据集改形状
    x_train = X_train.reshape(X_train.shape[0], 28 * 28)

    # mnist标签集转numpy
    x_label = train_loader.dataset.train_labels.numpy()
    # 归一化
    x_train = meanNorm(x_train, x_train)

    return x_train, x_label



# getMnistData()
# print(type(train_loader))
# for i, (images, labels) in enumerate(train_loader):  # train_loader 是一个批次的数据
#     print(images.shape)
#     images = Variable(images.view(-1, 28 * 28))
#     print(images.shape)
#     labels = Variable(labels)
#     print(np.array(images.data).shape[0])