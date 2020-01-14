# -- coding: utf-8 --
# !/bin/python
import os
import sys
import cv2
import uuid
from src.Vision.camera import Camera
from src.Vision.vision_duplication import *
from src.Vision.imageProcess.imgProc_duplication import ImgProc
import time
from timeit import default_timer as timer
import numpy as np
from src.Vision.yolo.Yolo import *


class Track:
    """
    根据图像api，提供增加新的Target目标功能；
    提供更新实时Target目标功能；

    """

    def createTarget(self, bottleDict):
        """
        增加新的Target目标功能

        :return: 新的带UUID的targetDict
        """

        # 创建新的target并标记uuid 返回给bottleDict
        targetDict = dict()
        targetList = list()
        uuIDList = list()
        nFrame = bottleDict.get("nFrame")
        bgTimeCost = bottleDict.get("bgTimeCost")
        timeCost = bottleDict.get("timeCost")
        frameTime = bottleDict.get("frameTime")
        targetTrackTime = 0

        targetDict.setdefault("target", targetList)

        for i in range(len(bottleDict.get("box"))):
            tempList = list()
            trackFlag = 0
            position = [int((bottleDict["box"][i][2] + bottleDict["box"][i][4]) / 2),
                        int((bottleDict["box"][i][3] + bottleDict["box"][i][5]) / 2)]
            speed = [50, 50]
            angle = 0
            type = 0
            typeCounter = 0

            uuID = str(uuid.uuid1())    # 自己创建，用uuid1 打上 UUID
            uuIDList.append(uuID)
            tempList.append(uuID)
            tempList.append(trackFlag)
            tempList.append(position)
            tempList.append(speed)
            tempList.append(angle)
            tempList.append(type)
            tempList.append(typeCounter)
            targetList.append(tempList)
        targetDict.setdefault("nFrame", nFrame)
        targetDict.setdefault("bgTimeCost", bgTimeCost)
        targetDict.setdefault("timeCost", timeCost)
        targetDict.setdefault("targetTrackTime", targetTrackTime)
        targetDict.setdefault("frameTime", frameTime)
        # tempList.append('\n')

        # file = open("targetDict_test.txt", "a")
        # for target in tempList:
        #     file.writelines(target + ", ")
        # file.writelines("\n")
        # print(targetDict, uuIDList)
        return targetDict, uuIDList

    def updateTarget(self, targetDict, _frame, flag = 0):
        """
        更新target功能，自定义时间间隔Δt = （t2 - t1），主流程会根据该时间间隔进行call bgLearn；
        :param targetDict: 上一步的目标物的信息
        :param _frame:
        :param flag: =0：表示更新在间隔内； =1：10次更行完成，置1
        :return: 同一UUID下的目标物的信息更新；
        """

        deltaT = 0.01
        self._frame = _frame
        oldTargetDict = targetDict
        newTargetDict = oldTargetDict
        frameTime = newTargetDict.get("frameTime")
        newTargetDictLists = oldTargetDict.get("target")
        # 循环遍历，更新target，
        for i in range(len(newTargetDictLists)):
            newTargetDictLists[i][2][0] = newTargetDictLists[i][2][0] + float(newTargetDictLists[i][3][0]) * (deltaT)
            newTargetDictLists[i][2][1] = newTargetDictLists[i][2][1] + float(newTargetDictLists[i][3][1]) * (deltaT)
            cv2.rectangle(_frame, (int(newTargetDictLists[i][2][0] - 50), int(newTargetDictLists[i][2][1] - 50)),
                          (int(newTargetDictLists[i][2][0]) + 100, int(newTargetDictLists[i][2][1]) + 100), (125, 0, 125), 4)
            # print(i)
        # targetTrackTime 更新为Δt后：
        newTargetDict["targetTrackTime"] = frameTime + (deltaT)
        # [a, b] = newTargetDict["target"][0][2]
        # cv2.rectangle(_frame, (int(a), int(b)), (int(a) + 50, int(b) + 50), (125, 0, 125), 4)
        # print("frameTime:" + str(newTargetDict["frameTime"]) + "     targetTrackTime:" + str(newTargetDict["targetTrackTime"])  + "     realTime:" + str(time.time()))
        return newTargetDict

    def mergeTarget(self, targetDict1, targetDict2):
        """
        合并target功能，实现实时运行中为实际镜头图像范围内的所有目标物
        :param targetDict1: 比较的新创立的target
        :param targetDict2: 比较的原先的在运行过程中的target
        :return: 合并后的target
        """
        # tempDict = targetDict2
        tempList = targetDict2.get("target")
        for i in range(len(targetDict1.get("target"))):
            tempList.append(targetDict1.get("target")[i])
        targetDict2.setdefault("target",tempList)
        # print(targetDict2)
        return 0

    def checkTarget(self, bottleDict):
        """
        检查target功能，自定义时间间隔Δt = （t2 - t1），主流程会根据该时间间隔进行call bgLearn；

        :param bottleDict: 上一步的
        :return: 同一UUID的信息更新；
        """

        # 将bottleDict中数据进行换算，并更新至targetDict内相对应的target

        file = open("targetDict_test.txt", "r+")

        # 逐行读取多行文件中的targetDict，与更新成UUID为相同一个的bottleDict中的值
        while True:
            tempList = file.readlines(10000)

            if not tempList:
                break
            for targetList in tempList:
                # 对比UUID ，假如一样则执行更新
                # 临时tempLists
                tempLists = bottleDict.get("target")
                print(tempLists)
                tempSingleList = targetList.split(", ")
                # print(tempLists)
                if (tempLists[0] in targetList):
                    # 赋值：给与每一个tempSingleList
                    file.write(tempLists[0] + ", ")
                    print(len(tempSingleList))
                    for i in range(6):
                        tempSingleList[i + 1] = tempLists[i + 1]
                        file.write(tempSingleList[i + 1] + ", ")
                    print(tempSingleList)

                    file.write("\n")
                break
                # print(targetList)
        file.close()


if __name__ == "__main__":
    # cam, _image = imageInit()
    cam = Camera()
    yolo = YOLO()
    _vision = Vision(cam, yolo, imgproc_=None)
    _imgproc = ImgProc(10)
    prev_time = timer()
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"

    # trackObj = ImageTrack()
    preframe, nFrame, t = cam.getImage()
    preframeb, bgMaskb, resarrayb = _imgproc.delBg(preframe) if _imgproc else (preframe, None)
    flag = 0
    inputCorner = np.array([])

    feature_params = dict(maxCorners=30,
                          qualityLevel=0.3,
                          minDistance=7,  # min distance between corners
                          blockSize=7)  # winsize of corner
    # params for lk track
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p0 = np.array([])
    label = np.array([])

    dataDict = dict()
    tempDict = dict()
    while True:
        frame, nFrame, t = cam.getImage()
        frame, bgMask, resarray = _imgproc.delBg(frame) if _imgproc else (frame, None)

        targetTracking = Track()

        # 图像识别出数据
        if "box" in dataDict:

            if "target" not in tempDict:
                tempDict, uuIDList = targetTracking.createTarget(dataDict)

            tempDict = targetTracking.updateTarget(tempDict, frame)  # 更新时，要求在targetTrackTime上做自加，这里updateTarget()还需改进！！
            print(tempDict)


        else:
            tempDict = dict()
            img = PImage.fromarray(frame)  # PImage: from PIL import Vision as PImage
            dataDict = _vision.yolo.detectImage(img)
            dataDict["bgTimeCost"] = _imgproc.bgTimeCost if _imgproc else 0
            result = np.asarray(dataDict["image"])
            # dataDict["image"] = result  # result：cv2.array的图像数据
            dataDict["image"] = img  # img：Image对象
            dataDict["nFrame"] = nFrame
            dataDict["frameTime"] = t  # 相机当前获取打当前帧nFrame的时间t
            print(tempDict)

        # tempDict, uuID = Track().createTarget(bottleDict)
        #
        # _frame, nFrame, t = cam.getImage()
        # tempDict["nFrame"] = nFrame
        #
        # # 虚拟间隔时间10s 增加targetDict，实际后续由vision中api提供
        # if tempDict is None and tempDict.get("frameTime") is not None:
        #     if tempT is None:
        #         tempT = 0
        #     tempT = tempT + t - tempDict.get("frameTime")
        #     # print(str(tempDict["frameTime"]) + ",   " + str(t) + ",   " + str(tempT))
        #     if tempT > 10:
        #         tempT = 0
        #         tempDict3, uuID2 = Track().createTarget(bottleDict)
        #         Track().mergeTarget(tempDict3, tempDict)
        #
        # tempDict["frameTime"] = t
        #
        # # 判断条件 还有待更改，这里只是调试本脚本示范用，Main中要重新改写
        # # if (tempDict["targetTrackTime"] == 0 or abs(t - tempDict["targetTrackTime"]) < 0.08 ):
        # tempDict = Track().updateTarget(tempDict)
        # print(str(tempDict["frameTime"]) + ",   " + str(t) + ",   " + str(tempDict["targetTrackTime"]) + ",   "+ str(time.time()))
        #
        cv2.imshow("test", frame)
        # tempImgproc = ImgProc(10)
        #
        # frame, bgMask, resarray = tempImgproc.delBg(_frame) if tempImgproc else (_frame, None)
        #
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.destroy()