import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import cv2
import torchvision.transforms as transforms
import operator

def knn_classify(k, dis, group, label, y_test):
    assert dis == "E" or dis == "M", "dis must E or M"
    num_test = y_test.shape[0]
    lablelist = []
    if dis == 'E':
        for i in range(num_test):
            #一个test data 和每个train data 都算一下距离  按列求和
            distances = np.sqrt(np.sum(((group - np.tile(y_test[i], (group.shape[0], 1))) ** 2), axis=1))
            nearest_k = np.argsort(distances)
            topK = nearest_k[:k]
            classCount = {}
            for j in topK:
                classCount[label[j]] = classCount.get(label[j], 0) + 1
            #items() 转换成key ,value的元组  label 数量
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            print("sortedClassCount", sortedClassCount)
            lablelist.append(sortedClassCount[0][0])
        return np.array(lablelist)
    if dis == 'M':
        for i in range(num_test):
            distances = np.sqrt(np.sum(abs((group - np.tile(y_test[i], (group.shape[0], 1)))), axis=1))
            nearest_k = np.argsort(distances)
            topK = nearest_k[:k]
            classCount = {}
            for j in topK:
                classCount[label[j]] = classCount.get(label[j], 0) + 1
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            print("sortedClassCount", sortedClassCount)
            lablelist.append(sortedClassCount[0][0])
        return np.array(lablelist)

#均值归一化处理
def meanNorm(dataForMean, inputData):
    mean_image = dataForMean.mean(axis=0)
    res = (inputData - mean_image) / 255
    return res