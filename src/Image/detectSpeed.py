import math
import threading
import time

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
    # 获取传送带速度，以 (Vx ,Vy. Vz)的格式返回，此格式可以代表速度的大小和方向

    # 实现方法：
    # 首先使用两个连续帧作为一组，得到它们的dataDict，找到两帧中同一类型的瓶子，而且该类瓶子在每帧中只有一个，

    # 获取第一帧中该瓶子的矩形框的中心点坐标（x1,y1），并将该点根据传送带和水平方向的夹角，
    # 映射到传送带方向上，得到（x1',y1'）
    # 获取第二帧中该瓶子的矩形框的中心点坐标（x2,y2），并将该点根据传送带和水平方向的夹角，
    # 映射到传送带方向上，得到（x2',y2'）
    # 计算出来（x1',y1'）和（x2',y2'）的欧氏距离，除以两帧之间的时间，得到速度，并分解到X,Y,Z 方向上
    # 得到(Vx ,Vy. Vz)，其中Vz默认是0
    # 然后再继续取10组连续帧，重复上述计算，每组得到得到一个(Vx ,Vy. Vz)，将这些(Vx ,Vy. Vz)都存入
    # 一个(Vx ,Vy. Vz)数组
    # 对这个(Vx ,Vy. Vz)数组，剔除过大和过小的异常数据，可以使用np.percentile方法，然后对剩余数据求平均获得
    # 最终（Vx ,Vy. Vz)
    """
    Parameters
    --------------
    I: input:
      dataDict

    Returns
    bottleDetail:
        mainly include bottle rotate angle from belt move direction,0--180 degree,
        and the diameter of bottle
    -------

    Examples
    --------
    """
    bottleDetail = []
    if not dataDict:
        print("未检测到物体！--1")
        return None
    # try get info of box in dataDict
    boxInfo = dataDict.get("box", 0)
    nFrame = dataDict["nFrame"]
    frameTime = dataDict["frameTime"]
    bottleDetail.append(nFrame)
    bottleDetail.append(frameTime)

    if not boxInfo:
        print("未检测到运动物体！--2")
        return None
    # get the first object which conditional more than 80%
    for i in boxInfo:

        if float(i[1]) > 0.8:
            bottleDetail.append(i[0])  # 瓶子类别
            # 换算成中心坐标
            centerX = (i[2] + i[4]) / 2
            centerY = (i[3] + i[5]) / 2
            bottleDetail.append((centerX, centerY))
            # bottleDetail.append(centerY)
            dstList.append(bottleDetail)
            break
        # print("dstList", dstList)
        # return dstList
            # test just one bottle
            # break
    if not dstList:
        print("未检测到运动物体！--3")
        return None
    print("检测到物体====>", dstList)
    return dstList


def getSpeed(dstList):
    speedList = []
    angleList = []
    for i in range(len(dstList)-1):
        if dstList[i][0] == dstList[i+1][0]:
            print("物体未运动，速度为0000--1")
            continue
        if dstList[i+1][3][0] == dstList[i][3][0] or dstList[i+1][3][1] == dstList[i][3][1]:
            print("物体未运动，速度为0000--2")
            continue
        sumTime = dstList[i+1][1] - dstList[i][1]
        pix_distance = math.sqrt((dstList[i+1][3][0] - dstList[i][3][0]) ** 2 + (dstList[i+1][3][1] - dstList[i][3][1]) ** 2)
        # 计算速度
        temSpeed = pix_distance / sumTime
        speedList.append(temSpeed)
        # 计算传送带运动速度与相机夹角
        radian = math.atan((dstList[i+1][3][1] - dstList[i][3][1]) / (dstList[i+1][3][0] - dstList[i][3][0]))
        # 将弧度转换成角度
        _angle = radian * 180 / math.pi
        angleList.append(_angle)
    # 计算十组的平均速度
    speed = sum(speedList) / len(speedList)
    # 计算传送带和相机角度
    angle = sum(angleList) / len(speedList)
    return speed, angle


def counter():
    for i in range(500):
        time.sleep(0.5)
        if not image.bottleDict:
            print(i, "未检测到物体！--4")
            continue
        print(i, image.bottleDict)
        dstList = getBeltSpeed(image.bottleDict)
        if not dstList:
            print("未检测到运动物体！--5")
            continue
        if len(dstList) == 20:
            # 计算速度
            speed, angle = getSpeed(dstList)
            print("*" * 50)
            print("speed", speed)
            print("angle", angle)
            print("*" * 50)
            break


if __name__ == '__main__':
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
    #

