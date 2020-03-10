import os
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import torchvision.models as models

from sampling import buildPredBoxes

__all__ = ["EzDetectConfig", "EzDetectNet", "ReorgModule"]

class EzDetectConfig(object):
    def __init__(self, batchSize=4, gpu=False):
        super(EzDetectConfig, self).__init__()
        self.batchSize = batchSize
        self.gpu = gpu
        self.classNumber = 21
        self.targetWidth = 330
        self.targetHeight = 330
        self.featureSize = [[42, 42],
                            [21, 21],
                            [11, 11],
                            [6, 6],
                            [3, 3]]
        #左边是大小系数 右边是比例  大于1代表瘦高的
        priorConfig = [[0.10, 0.25, 2],
                       [0.25, 0.4, 2, 3],
                       [0.4, 0.55, 2, 3],
                       [0.55, 0.7, 2, 3],
                       [0.7, 0.85, 2]]
        self.mboxes = []
        for i in range(len(priorConfig)):
            minSize = priorConfig[i][0]
            maxSize = priorConfig[i][1]
            meanSize = math.sqrt(minSize*maxSize)
            ratios = priorConfig[i][2:]
            #aspect ratio 1 for min and max
            self.mboxes.append([i, minSize, minSize])
            self.mboxes.append
