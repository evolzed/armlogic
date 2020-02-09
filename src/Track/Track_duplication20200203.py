# -- coding: utf-8 --
# !/bin/python
import os
import sys
import cv2
import uuid
from src.Vision.camera import Camera
from src.Vision.vision import *
from src.Vision.imageProcess.imgProc import *
import time
from timeit import default_timer as timer
import numpy as np
from src.Vision.yolo.Yolo import *
from multiprocessing import Process
import multiprocessing
from src.Track.kalmanFilter import KF

"""
from Track_duplication, start from 20200203
"""


class Track:
    """
    根据图像api，提供增加新的Target目标功能；
    提供更新实时Target目标功能；
    提供合并target功能，实现实时运行中为实际镜头图像范围内的所有目标物
    根据前一帧与当前帧的target信息，对target的速度进行估计计算
    提供总进程targetProc,

    """

    def createTarget(self, bottleDict,frame,featureimg,nFrame,_vision,_imgproc):
        """
        增加新的Target目标功能

        :return: 新的带UUID的targetDict, 传至imgProc的UUID
        """
        # 创建新的target并标记uuid 返回给bottleDict
        self.bottleDict = bottleDict
        # self.featureingimg = featureimg
        self.drawimg = frame

        img = PImage.fromarray(frame)  # PImage: from PIL import Vision as PImage
        # img.show()
        # feed data into model
        #神经网络识别  必须有
        dataDict = _vision.yolo.detectImage(img)

        dataDict["bgTimeCost"] = _imgproc.bgTimeCost if _imgproc else 0
        result = np.asarray(dataDict["image"])
        # dataDict["image"] = result  # result：cv2.array的图像数据
        dataDict["image"] = img  # img：Image对象
        # dataDict["timeCost"] = exec_time
        dataDict["nFrame"] = nFrame
        dataDict["frameTime"] = t  # 相机当前获取打当前帧nFrame的时间t


        targetDict = dict()
        targetList = list()
        uuIDList = list()
        nFrame = bottleDict.get("nFrame")
        bgTimeCost = bottleDict.get("bgTimeCost")
        timeCost = bottleDict.get("timeCost")
        frameTime = bottleDict.get("frameTime")
        targetTrackTime = frameTime

        targetDict.setdefault("target", targetList)

        p0, label, centerLists = _imgproc.detectObj(featureimg, frame, dataDict, 1)

        # 用imgProc的centerList
        # tempCenterList = _imgproc.findTrackedCenterPoint(p0, label)
        print("centerLists", centerLists)
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

        return targetDict, bottleDict, uuIDList, p0, label, centerLists

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


    def targetMove(self, x, y, last_measurement, last_prediction,current_measurement, current_prediction, kalman):

        # 定义全局变量
        global frame

        # 初始化
        last_measurement = current_measurement
        last_prediction = current_prediction
        # 传递当前测量坐标值
        current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
        kalman.correct(current_measurement)
        current_prediction = kalman.predict()
        trackFrame = np.zeros((1600, 1080, 3), np.uint8)

        # 上一次测量值
        lmx, lmy = last_measurement[0], last_measurement[1]
        cmx, cmy = current_measurement[0], current_measurement[1]
        lpx, lpy = last_prediction[0], last_prediction[1]
        cpx, cpy = current_prediction[0], current_prediction[1]
        cv2.line(trackFrame, (lmx, lmy), (cmx, cmy), (0, 100, 0))
        cv2.line(trackFrame, (lpx, lpy), (cpx, cpy), (0, 0, 200))

        print(lpx, cmx, cpx, lpy, cmy, cpy)


    def trackProcess(self, transDict, transList,):
        # kf1 = self.targetMove()
        kf2 = KF()
        kf3 = KF()
        kalman = cv2.KalmanFilter(4, 2)
        # 设置测量矩阵
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        # 设置转移矩阵
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # 设置过程噪声协方差矩阵
        kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        trackFrame = np.zeros((960, 1080, 3), np.uint8)
        # # 初始化测量坐标和target运动预测的数组
        last_measurement = current_measurement = np.array((2, 1), np.float32)
        last_prediction = current_prediction = np.zeros((2, 1), np.float32)

        flag = 0
        tempDict = dict()
        tempBottleDict = dict()

        # while len(transList) != 0:
        #     for i in range(len(transList)):
        #         print(transDict, transList, transList[i][0], transList[i][1])
        #     time.sleep(0.05)

        while True:
            print(transList, "-----------------------------------------")
            if transList:
                print(transList[0][0], transList[0][1],  "-----------------------------------------")
                self.targetMove(transList[0][0], transList[0][1],
                                last_measurement, last_prediction,
                                current_measurement, current_prediction, kalman)
            # cv2.imshow("kalman_tracker", trackFrame)

            print("*******")
            print(transDict)
            print(transList)
            print("*******")
            time.sleep(0.5)
        # cv2.destroyAllWindows()
            # if flag == 1:
            #     for i in range(10):
            #         # 更新target, 比较targetTrackTime 与最邻近帧时间，与其信息做比较：
            #         if i < 9:
            #             flag = 0
            #             tempDict = targetTracking.updateTarget(tempDict, time.time(), nFrame, flag, frame)
            #
            #         if i == 9:
            #             flag = 1
            #             # 更新时，要求在targetTrackTime上做自加，在最后一次子循环 update中问询imgProc.trackObj() 作判断
            #             tempDict = targetTracking.updateTarget(tempDict, time.time(), nFrame, flag, frame)

            # else:
            #     tempBottleDict = transDict
            #     """
            #     这里处理tempDict 和 tempBottleDict 之间关系
            #     """


if __name__ == "__main__":
    # cam, _image = imageInit()
    # # cam = Camera()
    # yolo = YOLO()
    #
    # videoDir = "d:\\1\\Video_20200204122301684.avi"
    # bgDir = "d:\\1\\背景1.avi"
    # avi = Video(videoDir)
    # bgAvi = Video(bgDir)
    # imgCapObj = imageCapture(None, avi, bgAvi)
    #
    # _imgproc = ImgProc(10, imgCapObj)
    # _imgproc.studyBackgroundFromCam(cam)
    # _imgproc.createModelsfromStats(6.0)
    # _vision = Vision(cam, yolo, _imgproc)
    # prev_time = timer()
    # accum_time = 0
    # curr_fps = 0
    # fps = "FPS: ??"
    # # trackObj = ImageTrack()
    # preframe, preframeb, _frame, frame = None, None, None, None
    # nFrame = 0
    # drawimg, featureimg, secondimg = None, None, None
    # flag = 0
    # inputCorner = np.array([])
    # p0 = np.array([])
    # label = np.array([])
    # dataDict = dict()
    # # tempDict = dict()
    # # feature_params = dict(maxCorners=30,
    # #                       qualityLevel=0.3,
    # #                       minDistance=7,  # min distance between corners
    # #                       blockSize=7)  # winsize of corner
    # # # params for lk track
    # # lk_params = dict(winSize=(15, 15),
    # #                  maxLevel=2,
    # #                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    #
    # preframe, nFrame, t = cam.getImage()
    # preframeb, bgMaskb, resarray = _imgproc.delBg(preframe) if _imgproc else (preframe, None)
    #
    # transDict = dict()
    #
    # pTarget = Process(target=Track.trackProcess, args=(transDict,))
    # pTarget.start()
    #
    # while True:
    #     targetTracking = Track()
    #
    #     #获取摄像机图片 进行去除背景 并灰度化  更新相机获取的图片
    #     _frame, nFrame, t = cam.getImage()
    #     frame, bgMaskb, resarrayb = _imgproc.delBg(_frame) if _imgproc else (_frame, None)
    #     featureimg = cv2.cvtColor(preframeb, cv2.COLOR_BGR2GRAY)
    #     secondimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     drawimg = frame.copy()
    #
    #
    #     # 图像识别出数据,赋予targetDict，在循环中作自身更新信息
    #     # if "box" in dataDict and flag == 1:
    #     if flag == 1:
    #         print("in track"*50)
    #         p0, label, centerList = _imgproc.trackObj(featureimg, secondimg, drawimg, label, p0)
    #         if centerList is not None and len(centerList) > 0:
    #             for seqN in range(len(centerList)):
    #                 cv2.circle(drawimg, (centerList[seqN][0], centerList[seqN][1]), 24, (255, 0, 0), 7)
    #         else:
    #             flag = 0
    #         """
    #         for i in range(10):
    #             # 更新target, 比较targetTrackTime 与最邻近帧时间，与其信息做比较：
    #             if i < 9:
    #                 flag = 0
    #                 tempDict = targetTracking.updateTarget(tempDict, time.time(), nFrame, flag, frame)
    #
    #             if i == 9:
    #                 flag = 1
    #                 # 更新时，要求在targetTrackTime上做自加，在最后一次子循环 update中问询imgProc.trackObj() 作判断
    #                 tempDict = targetTracking.updateTarget(tempDict, time.time(), nFrame, flag, frame)
    #         """
    #     else:
    #         # 创建target
    #         tempDict, tempBottleDict, uuIDList, p0, label, centerLists = targetTracking.createTarget(dataDict, frame, featureimg,nFrame,_vision, _imgproc)
    #         if centerLists is not None and len(centerLists) > 0:
    #             for seqN in range(len(centerLists)):
    #                 cv2.circle(drawimg, (centerLists[seqN][0], centerLists[seqN][1]), 24, (0, 0, 255), 7)
    #         if p0 is not None and label is not None:
    #             flag = 1
    #         else:
    #             flag = 0
    #         dataDict = tempBottleDict
    #         # 赋值通信transDict
    #         transDict = tempBottleDict
    #
    #     #updated preframb
    #     # 更新相机获取的 上一张图片
    #     preframeb = frame.copy()
    #     cv2.imshow("test", drawimg)
    #
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break
    # cam.destroy()
    track = Track()

    with multiprocessing.Manager() as MG:  # 重命名

        transDict = MG.dict()
        transList = MG.list()
        # cam, _image = imageInit()

        p2 = multiprocessing.Process(target=track.trackProcess, args=(transDict, transList))
        p2.daemon = True
        p2.start()
        # p2.join()

        p1 = multiprocessing.Process(target=vision_run, args=(transDict, transList, ))
        p1.daemon = True
        p1.start()
        p1.join()
        # _image.detectSerialImage(cam, transDict, )
