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
def numRecogByMnistKnn(num_pic, x_train, x_label, k):
    ret, num_pic = cv2.threshold(num_pic, 100, 255, cv2.THRESH_BINARY)
    mytest = 255 - cv2.cvtColor(num_pic, cv2.COLOR_BGR2GRAY)

    mytest = cv2.resize(mytest, (28, 28), interpolation=cv2.INTER_CUBIC)

    # cv2.imshow("my", mytest)

    #调整形状和数据集一致
    mytestData = mytest[np.newaxis, :]
    mytestData = mytestData.reshape(1, 28*28)

    #归一化保证和训练集特征尺度一致
    mytestData = meanNorm(x_train, mytestData)

    #用KNN识别
    pred = knn_classify(k, 'E', x_train, x_label, mytestData)
    print("pred0", pred)
    return pred[0]

def findTheNumPic(mytest0, left, top, w, h):
    #初始化参数
    kernel3 = np.ones((3, 3), np.uint8)
    show= mytest0.copy()
    #一步剪切出来大致位置
    mytest = cv2.cvtColor(mytest0[top: top + h, left: left + w], cv2.COLOR_BGR2GRAY)
    # cv2.imshow("mytest", mytest)
    cv2.rectangle(show, (left, top), (left+w, top+h), (0, 0, 255))
    #二值化然后膨胀然后边缘然后检测轮廓
    ret, threshed = cv2.threshold(mytest, 100, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("threshed", threshed)
    dilated = cv2.dilate(threshed, kernel3)

    # cv2.imshow("dilated", dilated)

    edge = cv2.Canny(dilated, 78, 148)

    # cv2.imshow("edge", edge)

    if cv2.__version__.startswith("3"):
        _, contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    elif cv2.__version__.startswith("4"):
        contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(show, contours, -1, (0, 255, 0), 1)
#找出轮廓面积最大的 就是数字图片
    arealist = []
    if len(contours) > 0:
        for ci in range(len(contours)):
            arclenth = cv2.arcLength(contours[ci], True)  # 面积
            area = cv2.contourArea(contours[ci])  # 4386.5
            arealist.append(area)
            print("area = ", area)
    print(arealist)
    sortIndex = sorted(range(len(arealist)), key=lambda k: arealist[k], reverse=True)
    print(sortIndex)

    # cv2.drawContours(show, contours, sortIndex[0], (0, 255, 255), 1)
    #找出最大的轮廓的外接矩形 并抠出来
    contourBndBox = cv2.boundingRect(contours[sortIndex[0]])  # x,y,w,h  外接矩形

    x = contourBndBox[0]+left
    y = contourBndBox[1]+top
    w = contourBndBox[2]
    h = contourBndBox[3]
    cv2.rectangle(show, (x, y), (x + w, y + h), (255, 255, 0), 2)  # 画矩形

    res = mytest0[y:y+h, x:x+w].copy()
    res = cv2.copyMakeBorder(res, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])#扩充边界


    # cv2.imshow("show", show)
    # cv2.imshow("res", res)
    return res, x, y, w, h
    # cv2.imshow("edge", edge)

if __name__ == "__main__":
    #读取待识别数字图片
    #6 7 9 识别有问题
    # mytest = cv2.imread("E:\\1\\1\\9_0.jpg")
    x_train, x_label = getMnistData()

    my = cv2.imread("E:\\1\\2\\1.jpg")

    left = 530
    top = 280
    w = 70
    h = 90
    numPic, x0, y0, w0, h0 = findTheNumPic(my, left, top, w, h)



    cv2.rectangle(my, (x0, y0), (x0 + w0, y0 + h0), (255, 255, 0), 2)  # 画矩形
    cv2.imshow("jpg", numPic)



    pred = numRecogByMnistKnn(numPic, x_train, x_label, 10)

    cv2.putText(my, text=str(pred), org=(x0, y0 - 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 255), thickness=2)

    cv2.imshow("1", my)

    # pred = numRecogByMnistKnn(mytest, x_train, x_label, 10)
    print("pred", pred)
    cv2.waitKey()