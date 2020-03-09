from os import listdir #用于解析VOC数据路径
from os.path import join
from random import random
from PIL import Image,  ImageDraw
import xml.etree.ElementTree #用于解析VOC的xmllabel

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

# from sampling import sampleEzDetect

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

#定义数据类 对数据进行预处理
class vocDataset(data.Dataset):
    def __init__(self, config, isTraining=True):
        super(vocDataset, self).__init__()
        self.isTraining = isTraining
        self.config = config
#使用均值和方差对图片的RGB值分别进行归一化，
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #toTensor先将输入归一化到(0, 1)
        self.transformer = transforms.Compose([transforms.ToTensor(), normalize])

    def __getitem__(self, index):
        item = None
        if self.isTraining:
            item = allTrainingData[index % len(allTrainingData)]
        else:
            item = allTestingData[index % len(allTestingData)]
        img = Image.open(item[0])   #item[0]为图像数据
        allBboxes = getVOCInfo(item[1])  #item[1]为通过getVOCInfo 函数解析出真实label的数据
        imgWidth, imgHeight = img.size

        targetWidth = int((random()*0.25+0.75)*imgWidth)
        targetHeight = int((random()*0.25+0.75)*imgHeight)

        #对图片进行随机crop ，并保证bbox的大小
        xmin = int( random()*(imgWidth - targetWidth) )
        ymin = int( random()*(imgWidth - targetHeight))
        img = img.crop((xmin, ymin, xmin+targetWidth, ymin+targetHeight))
        img = img.resize((self.config.targetWidth, self.config.targetHeight),Image.BILINEAR)
        imgT = self.transformer(img)
        imgT = imgT*256

        #调整bbox
        bboxes = []
        for i in allBboxes:
            xl = i['bbox'][0] - xmin  #左
            yt = i['bbox'][1] - ymin   #顶
            xr = i['bbox'][2] - xmin    #右
            yb = i['bbox'][3] - ymin   #底
            #防止超界
            if xl < 0:
                xl = 0
            if  xr >= targetWidth:
                xr = targetWidth - 1
            if  yt <0:
                yt =0
            if  yb >= targetHeight:
                yb = targetHeight - 1

            if (xr-xl>=0.05 and yb-yt >= 0.05):
                bbox = [ vocClassID[i['category_id']], xl, yt, xr, yb]
                bboxes.append(bbox)

            if len(bboxes) == 0:
                return self[index + 1]

            target = sampleEzDetect(self.config, bboxes)

            #对预测图片进行测试##
            draw = ImageDraw.Draw(img)
            num = int(target[0])
            for j in range(0,num):
                offset = j*6
                if (target[offset + 1]<0):
                    break
                k = int(target[offset + 6])
                trueBox = [target[offset+2],
                           target[offset+3],
                           target[offset+4],
                           target[offset+5]]
                predBox = self.config.predBoxes[k]
                draw.rectangle(trueBox[0]*self.config.targetWidth,
                               trueBox[1] * self.config.targetHeight,
                               trueBox[2] * self.config.targetWidth,
                               trueBox[3] * self.config.targetHeight,
                               )
                draw.rectangle(predBox[0]*self.config.targetWidth,
                               predBox[1] * self.config.targetHeight,
                               predBox[2] * self.config.targetWidth,
                               predBox[3] * self.config.targetHeight,
                None,"red")
            del draw
            img.save("/tmp/{}.jpg".format(index))
            return imgT,target

    def __len__(self):
        if self.isTraining:
            num = len(allTrainingData) - (len(allTrainingData) % self.config.batchSize)
            return num
        else:
            num = len(allTestingData) - (len(allTestingData) % self.config.batchSize)
            return num

#从VOC2007中读取数据


vocClassID = {}
#识别分类的名称转换成ID
for i in range(len(vocClassName)):
    vocClassID[vocClassName[i]] = i+1
print(vocClassID)


allTrainingData = []
allTestingData = []

#voc数据集存放路径
allFloder = ["E:\\ml\\voc2007\\VOCdevkit\\VOC2007"]
for floder in allFloder:
    print("floder", floder)
    imagePath = join(floder, "JPEGImages")
    infoPath = join(floder, "Annotations")
    index = 0
    # print(listdir(imagePath))
    for f in listdir(imagePath):
        if f.endswith(".jpg"):
            imageFile = join(imagePath, f)
            infoFile = join(infoPath, f[:-4]+".xml")
            if index % 10 == 0:
                #列表 文件加xml
                allTestingData.append((imageFile, infoFile))
            else:
                allTrainingData.append((imageFile, infoFile))
            index = index + 1
    print("allTestingData", allTestingData)























































