import math
import threading
import time

from lib.Logger.Logger import *
from src.Image.camera import Camera
from src.Image.image import Image
from src.Image import image
from src.Image.imageProcess.bgLearn import Bglearn
from src.Image.yolo.Yolo import YOLO
import sys, os

# print(sys.path)
# sys.path.append(os.path.abspath("../../../"))
# sys.path.insert(0, os.path.abspath("../../../src/Image/yolo"))
dstList = []  # use for store infomation of per box


def getBeltSpeed(dataDict):
    # 功能需求描述:
    # 获取传送带速度，以speed(Unit:pix/second), angle返回，此格式可以代表速度的大小和方向

    # 实现方法：
    # 1、 first use 20 frames as destination used，get their dataDicts，
    #   calculate speed use its time & position from dataDict and next frame
    # 2、transfer coordinate(xmin, ymin, xmax, ymax) to center coordinate(vX, vY)
    # 3、then calculate average speed get from step one

    """
    Parameters
    --------------
    dataDict：frame info-dataDict contains time、center coordinate

    Returns：speed，angle
    """
    global dstList
    bottleDetail = []
    # 获取dataDict中的box信息 | try get info of box in dataDict
    boxInfo = dataDict.get("box", 0)
    if not boxInfo:
        # print("未检测到运动物体！--2")
        return None
    # 获得第一个置信度大于80%的物体 | get the first object which conditional more than 80%
    for i in boxInfo:
        if float(i[1]) > 0.8:
            bottleDetail.append(dataDict["nFrame"])
            bottleDetail.append(dataDict["frameTime"])
            bottleDetail.append(i[0])  # 瓶子类别
            # 换算成中心坐标 | transfer to center coordinate
            centerX = (i[2] + i[4]) / 2
            centerY = (i[3] + i[5]) / 2
            bottleDetail.append((centerX, centerY))
            dstList.append(bottleDetail)
            break
    if not dstList:
        # print("未检测到运动物体！--3")
        return None
    # print("检测到物体====>", dstList)
    # print("采集目标物个数：", len(dstList))
    return dstList


def getSpeed(dstList):
    speedList = []
    angleList = []
    for i in range(len(dstList) - 1):
        if dstList[i][0] == dstList[i + 1][0]:
            # print("物体未运动，速度为0000--1")
            continue
        if dstList[i + 1][3][0] == dstList[i][3][0] and dstList[i + 1][3][1] == dstList[i][3][1]:
            # print("物体未运动，速度为0000--2")
            continue
        sumTime = dstList[i + 1][1] - dstList[i][1]
        # 得到x2-x1 和y2-y1的差值
        vX = dstList[i + 1][3][0] - dstList[i][3][0]
        vY = dstList[i + 1][3][1] - dstList[i][3][1]
        # 一旦vX或vY小于阈值，判断
        if 0 <= vX < 5 and 0 <= vY < 5:
            # print("小于阈值，物体静止--1")
            continue
        # 设置检测物体脱离视野的触发器
        if vX <= 0:
            # 一旦检测到i+1个物体坐标的x坐标小于i个物体的x坐标，舍弃该组信息，继续下一组运算
            # print("异常数据，跳过！！！！")
            continue
        pix_distance = math.sqrt(vX ** 2 + vY ** 2)
        # 计算速度
        temSpeed = pix_distance / sumTime
        speedList.append(temSpeed)
        # 计算传送带运动速度与相机夹角
        radian = math.atan(vY / vX)
        # 将弧度转换成角度
        _angle = radian * 180 / math.pi
        angleList.append(_angle)
    # 计算十组的平均速度
    if len(speedList):
        speed = sum(speedList) / len(speedList)
    else:
        speed = 0
    # 计算传送带和相机角度
    if len(angleList):
        angle = sum(angleList) / len(angleList)
    else:
        angle = 0
    print("speedLen===>{},angleLen===>{}".format(len(speedList), len(angleList)))
    return speed, angle


def counter():
    # for i in range(500):
    while True:
        time.sleep(0.2)
        if not image.bottleDict:
            # print("未检测到物体！--4")
            continue
        # print(image.bottleDict)
        retList = getBeltSpeed(image.bottleDict)
        if not retList:
            # print("未检测到运动物体！--5")
            continue
        if len(retList) == 20:
            # 计算速度
            speed, angle = getSpeed(retList)
            print(speed)
            speed = 0 if speed < 5 else speed
            print("*" * 50)
            print("speed", speed)
            print("angle", angle)
            print("*" * 50)
            # 置空dstList列表，重新开始
            global dstList
            dstList = []


if __name__ == '__main__':
    # sys.stdout = Logger("D:\\log1.txt")
    cam = Camera()
    yolo = YOLO()
    # bgobj = Bglearn(50)
    # bgobj.studyBackgroundFromCam(cam)
    # bgobj.createModelsfromStats(6.0)
    _image = Image(cam, yolo)

    # print("=" * 50)
    # print(bottleDict)
    # print("=" * 50)
    # counter()
    try:
        hThreadHandle = threading.Thread(target=counter, daemon=True)
        hThreadHandle.start()
    except:
        print("error: unable to start thread")

    _image.detectSerialImage(cam)
    # hThreadHandle.join()
    # # img.detectSerialImage(cam)
    # for i in range(500):
    #     time.sleep(1)
    #     if not bottleDict:
    #         print(i, "未检测到物体！")
    #         continue
    #     print(i, bottleDict)
    # getBeltSpeed(image.bottleDict)
    # # hThreadHandle.join()
    # # cam.destroy()
