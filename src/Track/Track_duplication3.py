# -- coding: utf-8 --
# !/bin/python
import os
import sys
import cv2
import uuid
from src.Vision.camera import Camera
from src.Vision.vision_duplication import *
from src.Vision.imageProcess.imgProc_duplication import *
import time
from timeit import default_timer as timer
import numpy as np
from src.Vision.yolo.Yolo import *


class Track:
    """
    根据图像api，提供增加新的Target目标功能；
    提供更新实时Target目标功能；
    提供合并target功能，实现实时运行中为实际镜头图像范围内的所有目标物
    根据前一帧与当前帧的target信息，对target的速度进行估计计算

    """

    def createTarget(self, bottleDict, frame):
        """
        增加新的Target目标功能

        :return: 新的带UUID的targetDict, 传至imgProc的UUID
        """

        # 创建新的target并标记uuid 返回给bottleDict
        self.bottleDict = bottleDict
        # self.featureingimg = featureimg
        self.drawimg = frame

        preframe, nFrame, t = cam.getImage()
        preframeb, bgMaskb, resarrayb = _imgproc.delBg(preframe) if _imgproc else (preframe, None)

        targetDict = dict()
        targetList = list()
        uuIDList = list()
        nFrame = bottleDict.get("nFrame")
        bgTimeCost = bottleDict.get("bgTimeCost")
        timeCost = bottleDict.get("timeCost")
        frameTime = bottleDict.get("frameTime")
        targetTrackTime = frameTime

        targetDict.setdefault("target", targetList)

        # if "box" not in bottleDict:
        _frame, nFrame, t = cam.getImage()
        frame2, bgMask, resarray = _imgproc.delBg(_frame) if _imgproc else (_frame, None)

        img = PImage.fromarray(frame2)  # PImage: from PIL import Vision as PImage
        bottleDict = _vision.yolo.detectImage(img)
        bottleDict["bgTimeCost"] = _imgproc.bgTimeCost if _imgproc else 0
        # result = np.asarray(dataDict["image"])
        # dataDict["image"] = result  # result：cv2.array的图像数据
        bottleDict["image"] = img  # img：Image对象
        bottleDict["nFrame"] = nFrame
        bottleDict["frameTime"] = t  # 相机当前获取打当前帧nFrame的时间t
        frame = frame2.copy()
        featureimg = cv2.cvtColor(preframeb, cv2.COLOR_BGR2GRAY)
        p0, label, centerpoints = _imgproc.detectObj(featureimg, frame, bottleDict, 3)

        # 用imgProc的centerList
        # tempCenterList = _imgproc.findTrackedCenterPoint(p0, label)
        print(centerpoints)
        print(p0, label)
        # cv2.imshow("test", frame)

        # else:
        if "box" in bottleDict:
            for i in range(len(bottleDict.get("box"))):
                tempList = list()
                trackFlag = 0
                position = [int((bottleDict["box"][i][2] + bottleDict["box"][i][4]) / 2),
                            int((bottleDict["box"][i][3] + bottleDict["box"][i][5]) / 2)]

                # 第一帧用传输带速度或估计值，后续用speedEstimate()!!
                speed = [5, 0]
                angle = 0
                type = 0
                typeCounter = 0

                # uuID = label
                uuID = str(uuid.uuid1())    # 先用label,  之后自己创建，用uuid1 打上 UUID

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
        cv2.imshow("test", frame)
        preframeb = frame.copy()
        return targetDict, bottleDict, uuIDList, preframeb

    def updateTarget(self, targetDict, _currentTime, _nFrame, flag, _frame=None):
        """
        更新target功能，自定义时间间隔Δt = （t2 - t1），主流程会根据该时间间隔进行call bgLearn；
        :param targetDict: 上一步的目标物的信息
        :param _frame:
        :param flag: =0：表示更新在间隔内； =1：10次更行完成，置1
        :return: 同一UUID下的目标物的信息更新；
        """

        self._nFrame = _nFrame
        self.flag = flag
        self._frame = _frame

        deltaT = 0.005
        oldTargetDict = targetDict
        newTargetDict = oldTargetDict
        startTime = _currentTime
        newTargetDictLists = oldTargetDict.get("target")
        print(flag)

        # if flag == 0:
            # 循环遍历，更新target
        for i in range(len(newTargetDictLists)):
            newTargetDictLists[i][2][0] = newTargetDictLists[i][2][0] + float(newTargetDictLists[i][3][0]) * (deltaT)
            newTargetDictLists[i][2][1] = newTargetDictLists[i][2][1] + float(newTargetDictLists[i][3][1]) * (deltaT)
            newTargetDictLists[i][1] = flag
            # 假如时间间隔到规定次数，则与imgProc中的trackObj进行对照，或其他后续待完善逻辑
            if flag == 1:
                _frame, nFrame, t = cam.getImage()
                frame, bgMask, resarray = _imgproc.delBg(_frame) if _imgproc else (_frame, None)
                drawimg = _frame.copy()
                featureimg = cv2.cvtColor(preframeb, cv2.COLOR_BGR2GRAY)
                secondimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.rectangle(_frame, (int(newTargetDictLists[i][2][0] - 50), int(newTargetDictLists[i][2][1] - 50)),
                              (int(newTargetDictLists[i][2][0]) + 100, int(newTargetDictLists[i][2][1]) + 100), (125, 0, 125), 4)
                # 调用trackObj
                # 需用dataDict，建议centerList ，label设为uuID 直接在dataDict内调用；
                # p0, label = _imgproc.trackObj(featureimg, secondimg, drawimg, label, p0, lk_params)
                cv2.imshow("test", _frame)
        # targetTrackTime 更新为Δt后：
        newTargetDict["targetTrackTime"] = startTime + deltaT
        newTargetDict["nFrame"] = _nFrame
        time.sleep((deltaT - 0.002))
        newTargetDict["timeCost"] = time.time()
        print(newTargetDict)
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
        targetDict2.setdefault("target", tempList)
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

    def speedEstimate(self, targetDict_1, targetDict_2):
        """
        根据前一帧与当前帧的target信息，对target的速度进行估计计算

        :param targetDict_1: 前一帧targetDict
        :param targetDict_2: 当前帧的targetDict
        :return: targetDict_2
        """
        targetDictList_1 = targetDict_1.get("target")
        targetDictList_2 = targetDict_2.get("target")
        tempDeltaT = targetDict_2.get("timeCost") - targetDict_1.get("timeCost")
        for i in range(len(targetDictList_1)):
            targetDictList_2[i][3][0] = (targetDictList_2[i][2][0] - targetDictList_1[i][2][0]) / tempDeltaT
            targetDictList_2[i][3][1] = (targetDictList_2[i][2][1] - targetDictList_1[i][2][1]) / tempDeltaT
        print(targetDict_2)

        return None


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
    preframe, preframeb, _frame, frame = None, None, None, None
    nFrame = 0
    drawimg, featureimg, secondimg = None, None, None
    flag = 0
    inputCorner = np.array([])
    p0 = np.array([])
    label = np.array([])
    dataDict = dict()
    tempDict = dict()
    # feature_params = dict(maxCorners=30,
    #                       qualityLevel=0.3,
    #                       minDistance=7,  # min distance between corners
    #                       blockSize=7)  # winsize of corner
    # # params for lk track
    # lk_params = dict(winSize=(15, 15),
    #                  maxLevel=2,
    #                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        targetTracking = Track()
        # 图像识别出数据,赋予targetDict，在循环中作自身更新信息
        if "box" in dataDict:
            for i in range(10):
                # 更新target, 比较targetTrackTime 与最邻近帧时间，与其信息做比较：
                if i < 9:
                    flag = 0
                    tempDict = targetTracking.updateTarget(tempDict, time.time(), nFrame, flag, frame)

                if i == 9:
                    flag = 1
                    # 更新时，要求在targetTrackTime上做自加，在最后一次子循环 update中问询imgProc.trackObj() 作判断
                    tempDict = targetTracking.updateTarget(tempDict, time.time(), nFrame, flag, frame)

        else:
            # 创建target
            tempDict, tempBottleDict, uuIDList, preframeb = targetTracking.createTarget(dataDict, drawimg)
            dataDict = tempBottleDict

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.destroy()
