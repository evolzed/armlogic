from os import listdir #用于解析VOC数据路径
from os.path import join
from random import random
from PIL import Image,  ImageDraw
import xml.etree.ElementTree #用于解析VOC的xmllabel

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from sampling import sampleEzDetect

__all__ = ["vocClassName", "vocClassID", "vocDataset"]

vocClassName = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant", #盆栽
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

def getVOCInfo(xmlFile):
    root = xml.etree.ElementTree.parse(xmlFile).getroot()
    anns = root.findall("object")
    bboxes = []
    for ann in anns:
        name = ann.find('name').text
        newAnn = {}
        newAnn['category_id'] = name
        bbox = ann.find("bndbox")
        newAnn['bbox'] = [-1, -1, -1, -1]
        newAnn['bbox'][0] = float(bbox.find('xmin').text)
        newAnn['bbox'][1] = float(bbox.find('ymin').text)
        newAnn['bbox'][2] = float(bbox.find('xmax').text)
        newAnn['bbox'][3] = float(bbox.find('ymax').text)
        bboxes.append(newAnn)
    return bboxes

class vocDataset(data.Dataset):
    def __init__(self, config, isTraining=True):
        super(vocDataset,self).__init__()
        self.isTraining = isTraining
        self.config = config
#使用均值和方差对图片的RGB值分别进行归一化，
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        self.transformer = transforms.Compose([transforms.ToTensor(), normalize])

    def __getitem__(self,index):
        item = None
        if self.isTraining:













