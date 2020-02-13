import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import cv2
import torchvision.transforms as transforms
import operator

from tools.mnist import *
from tools.KNN import *

#通过mnist数据集和KNN算法来识别数字图片中的数字
def numRecogByMnistKnn(num_pic, x_train, x_label):

    mytest = 255 - cv2.cvtColor(num_pic, cv2.COLOR_BGR2GRAY)

    mytest = cv2.resize(mytest, (28, 28), interpolation=cv2.INTER_CUBIC)

    cv2.imshow("my", mytest)

    #调整形状和数据集一致
    mytestData = mytest[np.newaxis, :]
    mytestData = mytestData.reshape(1, 28*28)

    #归一化保证和训练集特征尺度一致
    mytestData = meanNorm(x_train, mytestData)

    #用KNN识别
    pred = knn_classify(5, 'E', x_train, x_label, mytestData)
    print("pred0", pred)
    return pred[0]


if __name__ == "__main__":
    #读取待识别数字图片
    mytest = cv2.imread("E:\\1\\1\\3_0.jpg")
    x_train, x_label = getMnistData()
    pred = numRecogByMnistKnn(mytest, x_train, x_label)
    print("pred", pred)
