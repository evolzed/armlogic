# -- coding: utf-8 --
# !/bin/python
import os
import sys
import cv2
import uuid
from src.BS02.camera import Camera

from src.BS02.vision import *
from src.BS02.imageProcess.imgProc import *
import time
from timeit import default_timer as timer
import numpy as np
from src.BS02.yolo.Yolo import YOLO
from multiprocessing import Process
import multiprocessing
from lib.Logger.Logger import Logger
"""
from Track_duplication, start from 20200203
"""
# sys.stdout = Logger("d:\\12.txt")  # 保存到D盘

speedMonitorList = [[]]
preVX = preVY = 0
cnt = 0


class Track:
    """
    根据图像api，提供增加新的Target目标功能；
    提供更新实时Target目标功能；
    提供合并target功能，实现实时运行中为实际镜头图像范围内的所有目标物
    根据前一帧与当前帧的target信息，对target的速度进行估计计算
    提供总进程targetProc,
    """

    # def createTarget(self, bottleDict,frame,featureimg,nFrame,_vision,_imgproc):
    def createTarget(self, transDict, transList):
        """
        增加新的Target目标功能
        :return: 新的带UUID的targetDict, 传至imgProc的UUID
        """
        # # 创建新的target并标记uuid 返回给bottleDict
        # self.bottleDict = bottleDict
        # # self.featureingimg = featureimg
        # self.drawimg = frame
        #
        # img = PImage.fromarray(frame)  # PImage: from PIL import Vision as PImage
        # # img.show()
        # # feed data into model
        # #神经网络识别  必须有
        # dataDict = _vision.yolo.detectImage(img)
        #
        # dataDict["bgTimeCost"] = _imgproc.bgTimeCost if _imgproc else 0
        # result = np.asarray(dataDict["image"])
        # # dataDict["image"] = result  # result：cv2.array的图像数据
        # dataDict["image"] = img  # img：Image对象
        # # dataDict["timeCost"] = exec_time
        # dataDict["nFrame"] = nFrame
        # dataDict["frameTime"] = t  # 相机当前获取打当前帧nFrame的时间t
        #
        #
        # targetDict = dict()
        # targetList = list()
        # uuIDList = list()
        # nFrame = bottleDict.get("nFrame")
        # bgTimeCost = bottleDict.get("bgTimeCost")
        # timeCost = bottleDict.get("timeCost")
        # frameTime = bottleDict.get("frameTime")
        # targetTrackTime = frameTime
        #
        # targetDict.setdefault("target", targetList)
        #
        # p0, label, centerLists = _imgproc.detectObj(featureimg, frame, dataDict, 1)
        #
        # # 用imgProc的centerList
        # # tempCenterList = _imgproc.findTrackedCenterPoint(p0, label)
        # print("centerLists", centerLists)
        # print(p0, label)
        # # cv2.imshow("test", frame)
        #
        # # else:
        # if "box" in bottleDict:
        #     for i in range(len(bottleDict.get("box"))):
        #         tempList = list()
        #         trackFlag = 0
        #         position = [int((bottleDict["box"][i][2] + bottleDict["box"][i][4]) / 2),
        #                     int((bottleDict["box"][i][3] + bottleDict["box"][i][5]) / 2)]
        #
        #         # 第一帧用传输带速度或估计值，后续用speedEstimate()!!
        #         speed = [5, 0]
        #         angle = 0
        #         type = 0
        #         typeCounter = 0
        #
        #         # uuID = label
        #         uuID = str(uuid.uuid1())    # 先用label,  之后自己创建，用uuid1 打上 UUID
        #
        #         uuIDList.append(uuID)
        #         tempList.append(uuID)
        #         tempList.append(trackFlag)
        #         tempList.append(position)
        #         tempList.append(speed)
        #         tempList.append(angle)
        #         tempList.append(type)
        #         tempList.append(typeCounter)
        #         targetList.append(tempList)
        #     targetDict.setdefault("nFrame", nFrame)
        #     targetDict.setdefault("bgTimeCost", bgTimeCost)
        #     targetDict.setdefault("timeCost", timeCost)
        #     targetDict.setdefault("targetTrackTime", targetTrackTime)
        #     targetDict.setdefault("frameTime", frameTime)
        #     # tempList.append('\n')
        #
        #     # file = open("targetDict_test.txt", "a")
        #     # for target in tempList:
        #     #     file.writelines(target + ", ")
        #     # file.writelines("\n")
        #     # print(targetDict, uuIDList)
        #
        # return targetDict, bottleDict, uuIDList, p0, label, centerLists
        startTime = time.time()
        deltaT = 0.01

        global trackFlag
        # trackFlag = 0  # 对应imgProc中的Flag

        targetDict = dict()
        speed = [0, 0]
        # speed = [24.83, 0]    # 设置初始速度，为录像视频中平均速度; 初始像素速度为5.8， 转换成时间相关速度 近似值#
        angle = [0, 0]
        type = 0
        typCounter = 0
        time.sleep(0.001)
        print(transList,  "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(len(transList) != 0, "##############")
        print(transList,  "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        tempList = [[] for j in range(len(transList))]
        # if transList:
        if len(transList) != 0:
            # tempList = [[] for j in range(len(transList))]
            print(tempList)
            for i in range(len(transList)):
                print(i)
                tempList[i].append(str(uuid.uuid1()))   # 对应位置打上uuID
                # tempList[i].append(str(i) + "**********")    # 测试用
                tempList[i].append(trackFlag)
                tempList[i].append([transList[i][0], transList[i][1]])
                tempList[i].append(speed)
                tempList[i].append(angle)
                tempList[i].append(type)
                tempList[i].append(typCounter)
            time.sleep(deltaT - 0.0025)    # 实际让程序运行总体控制在0.01s内；
        targetDict.setdefault("target", tempList)

        # 增加timeCost  和  targetTrackTime。。。
        timeCost = time.time()
        targetTrackTime = startTime + deltaT
        targetDict.setdefault("timeCost", timeCost)
        targetDict.setdefault("targetTrackTime", targetTrackTime)
        if len(tempList) != 0:
            trackFlag = 1
        print(targetDict, "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

        return targetDict

    def updateTarget(self, targetDict, transList):
        """
        更新target功能，自定义时间间隔Δt = （t2 - t1），主流程会根据该时间间隔进行call bgLearn；
        :param targetDict: 上一步的目标物的信息
        :param _frame:
        :param flag: =0：表示更新在间隔内； =1：10次更行完成，置1
        :return: 同一UUID下的目标物的信息更新；
        """

        # self._nFrame = _nFrame
        # self.flag = flag
        # self._frame = _frame
        #
        # deltaT = 0.005
        # oldTargetDict = targetDict
        # newTargetDict = oldTargetDict
        # startTime = _currentTime
        # newTargetDictLists = oldTargetDict.get("target")
        # print(flag)
        #
        # # if flag == 0:
        #     # 循环遍历，更新target
        # for i in range(len(newTargetDictLists)):
        #     newTargetDictLists[i][2][0] = newTargetDictLists[i][2][0] + float(newTargetDictLists[i][3][0]) * (deltaT)
        #     newTargetDictLists[i][2][1] = newTargetDictLists[i][2][1] + float(newTargetDictLists[i][3][1]) * (deltaT)
        #     newTargetDictLists[i][1] = flag
        #     # 假如时间间隔到规定次数，则与imgProc中的trackObj进行对照，或其他后续待完善逻辑
        #     if flag == 1:
        #         _frame, nFrame, t = cam.getImage()
        #         frame, bgMask, resarray = _imgproc.delBg(_frame) if _imgproc else (_frame, None)
        #         drawimg = _frame.copy()
        #         featureimg = cv2.cvtColor(preframeb, cv2.COLOR_BGR2GRAY)
        #         secondimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #         cv2.rectangle(_frame, (int(newTargetDictLists[i][2][0] - 50), int(newTargetDictLists[i][2][1] - 50)),
        #                       (int(newTargetDictLists[i][2][0]) + 100, int(newTargetDictLists[i][2][1]) + 100),
        #                       (125, 0, 125), 4)
        #         # 调用trackObj
        #         # 需用dataDict，建议centerList ，label设为uuID 直接在dataDict内调用；
        #         # p0, label = _imgproc.trackObj(featureimg, secondimg, drawimg, label, p0, lk_params)
        #         cv2.imshow("test", _frame)
        # # targetTrackTime 更新为Δt后：
        # newTargetDict["targetTrackTime"] = startTime + deltaT
        # newTargetDict["nFrame"] = _nFrame
        # time.sleep((deltaT - 0.002))
        # newTargetDict["timeCost"] = time.time()
        # print(newTargetDict)
        # return newTargetDict

        print(transList)
        global trackFlag
        startTime = time.time()
        deltaT = 0.01
        # tempTargetDict = dict()
        tempTargetDict = targetDict  # updateTarget()进程中，每次存储的临时TargetDict # ；

        # 每步updateTarget()进行自主估值
        # 从这里开始 tempList自 transList进行获取！
        tempList = transList
        # tempList = tempTargetDict["target"]
        for i in range(len(tempTargetDict["target"])):
            if len(tempTargetDict["target"]) != 0:
                # tempList[i][0] = tempList[i][0] + tempList[i][3] * deltaT
                # tempList[i][1] = tempList[i][1] + tempList[i][4] * deltaT
                # tempTargetDict["target"][i][2][0] = tempList[i][0] + tempList[i][3] * deltaT
                # tempTargetDict["target"][i][2][1] = tempList[i][1] + tempList[i][4] * deltaT
                tempTargetDict["target"][i][2][0] = tempTargetDict["target"][i][2][0] +\
                                                    tempTargetDict["target"][i][3][0] * deltaT
                tempTargetDict["target"][i][2][1] = tempTargetDict["target"][i][2][1] +\
                                                    tempTargetDict["target"][i][3][1] * deltaT
        # 回传targetDict
        targetDict = tempTargetDict
                # if len(tempList[i]) != 0:
            #     tempList[i][2][0] = tempList[i][2][0] + tempList[i][3][0] * deltaT
            #     tempList[i][2][1] = tempList[i][2][1] + tempList[i][3][1] * deltaT
        # tempTargetDict["target"] = tempList

        print(tempTargetDict, "this is tempDict !!!!!!")

        # targetDict.update(targetDict)    # 自主更新targetDict
        targetDict.update(tempTargetDict)    # 更新成targetDict
        time.sleep(deltaT - 0.0025)     # 实际让程序运行总体控制在0.01s内；

        timeCost = time.time()
        targetTrackTime = startTime + deltaT
        targetDict["timeCost"] = timeCost
        targetDict["targetTrackTime"] = targetTrackTime

        print(targetDict, transList, "^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        return targetDict

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
        print("$$$$$$$$$$$$$$$", targetDict2)
        return targetDict2

    def checkTarget(self, transDict, targetDict, transList):
        """
        检查target功能，在trackProcess循环末期，对照自身target信息和bottleDict中的centerList信息；
        :param bottleDict: 上一步的
        :return: 同一UUID的信息更新；
        """

        # # 将bottleDict中数据进行换算，并更新至targetDict内相对应的target
        #
        # file = open("targetDict_test.txt", "r+")
        #
        # # 逐行读取多行文件中的targetDict，与更新成UUID为相同一个的bottleDict中的值
        # while True:
        #     tempList = file.readlines(10000)
        #
        #     if not tempList:
        #         break
        #     for targetList in tempList:
        #         # 对比UUID ，假如一样则执行更新
        #         # 临时tempLists
        #         tempLists = bottleDict.get("target")
        #         print(tempLists)
        #         tempSingleList = targetList.split(", ")
        #         # print(tempLists)
        #         if (tempLists[0] in targetList):
        #             # 赋值：给与每一个tempSingleList
        #             file.write(tempLists[0] + ", ")
        #             print(len(tempSingleList))
        #             for i in range(6):
        #                 tempSingleList[i + 1] = tempLists[i + 1]
        #                 file.write(tempSingleList[i + 1] + ", ")
        #             print(tempSingleList)
        #
        #             file.write("\n")
        #         break
        #         # print(targetList)
        # file.close()
        k = 24.83 / 5.8
        # 目前 按照顺序依次对targetDict 中的list 与 transList 进行比较， 目前，对于循环末期，直接赋值transList
        # tempList为初始时target的List；


        # compare机制，用临时存储tempList，比较其长度与targetDict长度，假如追踪target在判断范围内，则不做修正，
        print("%"*150)
        print(targetDict, transList)
        print("%" * 150)

        tempTransList = transList
        if len(tempTransList) > len(targetDict["target"]):
            for i in range(len(tempTransList)):
                for ii in range(len(targetDict["target"])):
                    deltaX = tempTransList[i][0] - targetDict["target"][ii][2][0]
                    deltaY = tempTransList[i][1] - targetDict["target"][ii][2][1]
                    # 暂定范围
                    if deltaX * deltaX + deltaY * deltaY < 50:
                        tempTransList.pop(i)
            tempTargetDict = self.createTarget(transDict, tempTransList)
            targetDict = self.mergeTarget(targetDict, tempTargetDict)




        # tempList = targetDict["target"]
        # if len(tempList) > 0:
        #     for j in range(len(tempList)):
        #         # 位置直接赋值
        #         # 利用传送带方向的目标位置做判断transList是否及时更新，若没来得及更新，则targetDict延续自身位置信息；
        #         # 做判断，赋值
        #         # if transList[j][0] > tempList[j][2][0]:
        #         #
        #         #     tempList[j][2][0] = transList[j][0]
        #         #     tempList[j][2][1] = transList[j][1]
        #         # 不做判断直接赋值！
        #         tempList[j][2][0] = transList[j][0]
        #         tempList[j][2][1] = transList[j][1]
        #         tempList[j][3][0] = transList[j][3] * k
        #         tempList[j][3][1] = transList[j][4] * k  # 速度直接赋值  #
        # targetDict["target"] = tempList



        # 检查视野中的centerlist， 若空了，则清除targetDict （暂时处理）

        # if len(transList) == 0:
        # # 对于后续centerList 元素在相机视野中的位置等信息判断；
        # # judge = 1
        # # for l in range(len(transList)):
        # #     judge = (transList[l][0] < 125) * judge
        # # if judge == 1:
        #     targetDict["target"] = []
        targetDict.update(targetDict)
        print("@@@@@@@@@", targetDict, "@@@@@@@@@", str(len(transList)))

        return targetDict

    def speedEstimate(self, transDict, transList, targetDict):
        """
        根据前一帧与当前帧的centerList信息，对target的速度进行估计计算, 一旦监测到centerList对应position发生变化，
        则对两者之间的速度做出计算，一定要以变化的时候的时刻作为速度的计算依据；
        :param transDict:
        :param transList:
        :return:
        """

        tempList = transList
        # 记录当前时刻
        tempT = time.time()

        # targetDictList_1 = targetDict_1.get("target")
        # targetDictList_2 = targetDict_2.get("target")
        # tempDeltaT = targetDict_2.get("timeCost") - targetDict_1.get("timeCost")
        # for i in range(len(targetDictList_1)):
        #     targetDictList_2[i][3][0] = (targetDictList_2[i][2][0] - targetDictList_1[i][2][0]) / tempDeltaT
        #     targetDictList_2[i][3][1] = (targetDictList_2[i][2][1] - targetDictList_1[i][2][1]) / tempDeltaT
        # print(targetDict_2)

        return None

    def targetMove(self, x, y, current_measurement, last_measurement,
                   current_prediction, last_prediction, kalman, frame):

        """
        调用kalman滤波，将bottleDict,centerlist 信息作为检测数据； 被updateTarget（）所调用
        :param x:
        :param y:
        :param last_measurement:
        :param last_prediction:
        :param current_measurement:
        :param current_prediction:
        :param kalman:
        :return:
        """

        # # 定义全局变量
        # global frame

        # 初始化
        last_measurement = current_measurement
        last_prediction = current_prediction
        # 传递当前测量坐标值
        current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
        kalman.correct(current_measurement)
        current_prediction = kalman.predict()

        # 上一次测量值
        lmx, lmy = last_measurement[0], last_measurement[1]
        cmx, cmy = current_measurement[0], current_measurement[1]
        lpx, lpy = last_prediction[0], last_prediction[1]
        cpx, cpy = current_prediction[0], current_prediction[1]
        # cv2.line(frame, (lmx, lmy), (cmx, cmy), (0, 100, 0))
        # cv2.line(frame, (lpx, lpy), (cpx, cpy), (0, 0, 200))
        cv2.circle(frame, (cmx, cmy), 12, (0, 100, 0), 2)
        cv2.circle(frame, (cpx, cpy), 6, (0, 0, 200), 2)
        cv2.putText(frame, "green is current measurement", (cmx - 100, cmy - 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "red is current prediction", (cmx + 100, cmy), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.imshow("kalman_tracker", frame)
        print(lpx, lmx, cmx, cpx, lpy, lmy, cmy, cpy)

        #
        # # 定义全局变量
        # global frame
        #
        # # 初始化
        # for i in range(len(current_measurement)):
        #     last_measurement[i] = current_measurement[i]
        #     last_prediction[i] = current_prediction[i]
        # # 传递当前测量坐标值
        # current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
        # kalman.correct(current_measurement)
        # current_prediction = kalman.predict()
        # trackFrame = np.zeros((1600, 1080, 3), np.uint8)
        #
        # # 上一次测量值
        # lmx, lmy = last_measurement[0], last_measurement[1]
        # cmx, cmy = current_measurement[0], current_measurement[1]
        # lpx, lpy = last_prediction[0], last_prediction[1]
        # cpx, cpy = current_prediction[0], current_prediction[1]
        # cv2.imshow("kalman_tracker", trackFrame)
        # cv2.line(trackFrame, (lmx, lmy), (cmx, cmy), (0, 100, 0))
        # cv2.line(trackFrame, (lpx, lpy), (cpx, cpy), (0, 0, 200))
        #
        # print(lmx, lmy, cmx, cmy, "  &&&   ", lpx, lpy, cpx, cpy)

    def trackProcess(self, transDict, transList, targetDict):
        # # kf1 = self.targetMove()
        # kf2 = KF()
        # kf3 = KF()
        # kalman = cv2.KalmanFilter(4, 2)
        # # 设置测量矩阵
        # kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        # # 设置转移矩阵
        # kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # # 设置过程噪声协方差矩阵
        # kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        # trackFrame = np.zeros((960, 1080, 3), np.uint8)
        # # # 初始化测量坐标和target运动预测的数组
        # last_measurement = current_measurement = np.array((2, 1), np.float32)
        # last_prediction = current_prediction = np.zeros((2, 1), np.float32)
        #
        # flag = 0
        # tempDict = dict()
        # tempBottleDict = dict()
        #
        # # while len(transList) != 0:
        # #     for i in range(len(transList)):
        # #         print(transDict, transList, transList[i][0], transList[i][1])
        # #     time.sleep(0.05)
        #
        # while True:
        #     print(transList, "-----------------------------------------")
        #     if transList:
        #         print(transList[0][0], transList[0][1],  "-----------------------------------------")
        #         self.targetMove(transList[0][0], transList[0][1],
        #                         last_measurement, last_prediction,
        #                         current_measurement, current_prediction, kalman)
        #     lmx, lmy = int(last_measurement[0]), int(last_measurement[1])
        #     cmx, cmy = int(current_measurement[0]), int(current_measurement[1])
        #     lpx, lpy = int(last_prediction[0]), int(last_prediction[1])
        #     cpx, cpy = int(current_prediction[0]), int(current_prediction[1])
        #     # cv2.imshow("kalman_tracker", trackFrame)
        #     # cv2.line(trackFrame, (lmx, lmy), (cmx, cmy), (0, 100, 0))
        #     # cv2.line(trackFrame, (lpx, lpy), (cpx, cpy), (0, 0, 200))
        #     # cv2.line(trackFrame, (last_measurement[0], last_measurement[1]), (current_measurement[0], current_measurement[1]), (0, 100, 0))
        #     # cv2.line(trackFrame, (last_prediction[0], last_prediction[1]), (current_prediction[0], current_prediction[1]), (0, 0, 200))
        #     print("*******")
        #     print(transDict)
        #     print(lmx, lmy, cmx, cmy,"  &&&   ", lpx, lpy, cpx, cpy)
        #     print(transList)
        #     print("*******")
        #     time.sleep(0.5)
        #     if (cv2.waitKey(30) & 0xff) == 27:
        #         break
        # cv2.destroyAllWindows()
        global trackFlag, speedMonitorList, cnt, preVX, preVY
        trackFlag = 0

        # 存储上一记录targetDict
        lastDict = targetDict

        while True:

            trackFrame = np.zeros((960, 1280, 3), np.uint8)
            # if trackFlag == 1:  # 条件待定，与imgProc中的Flag信号， 以及需结合有没有产生新target信号结合
            print("*" * 200)
            print(transList)
            print(targetDict)
            print("*" * 200)
            if len(transList) != 0:
                for i in range(10):
                    # 更新target, 比较targetTrackTime 与最邻近帧时间，与其信息做比较：
                    if i < 9:
                        if len(targetDict) == 0:
                            targetDict= self.updateTarget(self.createTarget(transDict, transList, ), transList)
                        else:
                            targetDict = self.updateTarget(targetDict, transList)

                        if len(transList) != 0:
                            print(preVX == transList[0][0])
                            print(preVY == transList[0][1])

                    if i == 9:
                        # 更新时，要求在targetTrackTime上做自加，在最后一次子循环 update中问询imgProc.trackObj() 作判断

                        # # 目前 按照顺序依次对targetDict 中的list 与 transList 进行比较， 目前，对于循环末期，直接赋值transList
                        # tempList = targetDict["target"]
                        # for j in range(len(tempList) - 1):
                        #     # 位置直接赋值  利用传送带方向的目标位置做判断transList是否及时更新，若没来得及更新，则targetDict延续自身位置信息；
                        #     if transList[j][0] > tempList[j][2][0]:
                        #         tempList[j][2][0] = transList[j][0]
                        #         tempList[j][2][1] = transList[j][1]
                        #     tempList[j][3][0] = transList[j][3] * k
                        #     tempList[j][3][1] = transList[j][4] * k     # 速度直接赋值  #
                        #
                        # targetDict["target"] = tempList

                        targetDict = self.checkTarget(transDict, targetDict, transList,)
                        print(targetDict, transList, "^^^^^^^^^^^^^^^^^^^^^^^^^^^  this is loop end !!!")
                        # self.updateTarget(targetDict, transList)


                # test monitor!!!
                print("~" * 100)
                # if len(speedMonitorList) != 0:
                #     print(speedMonitorList, transList)
                #     # print(speedMonitorList[0][0] == transList[0][0])
                #     print(preV)
                #     print(transList[0][0])
                #     print("!!!!!", preV == transList[0][0])
                # print("~" * 100)
                # # 监控速度变化的机制：
                # cnt += 1

                # speedMonitorList = [[]for ii in range(transList)]
                if len(transList) > 0:
                    preVX = transList[0][0]
                    preVY = transList[0][1]
                    # speedMonitorList[0][0] = transList[0][0]

            else:
                time.sleep(0.001)
                # targetDict.update(self.createTarget(transDict, transList,))
                # # targetDict = self.createTarget(transDict, transList,)
                # print(targetDict, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")




            # show
            if len(transList) != 0 and len(targetDict) != 0:
                for j in range(len(targetDict["target"])):
                    currentX, currentY = int(targetDict["target"][j][2][0]), int(targetDict["target"][j][2][1])
                    # FilterShow
                    # lastX, lastY = int(lastDict["target"][j][2][0]), int(lastDict["target"][j][2][1],)
                    uuIDText = targetDict["target"][j][0]
                    cv2.circle(trackFrame, (currentX, currentY), 6, (0, 0, 200), 2)
                    cv2.putText(trackFrame, uuIDText, (currentX - 100, currentY + 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("tragetTracking", trackFrame)
            if (cv2.waitKey(30) & 0xff) == 27:
                break

    def trackWithFilter(self, transDict, transList, targetDict):
        x, y = 0, 0

        kalman = dict()

        # # 初始化测量坐标和target运动预测的数组; 定义要追踪的点参数
        last_measurement = current_measurement = last_prediction = current_prediction = []
        for l in range(100):
            last_measurement.append(np.array((2, 1), np.float32))
            current_measurement.append(np.array((2, 1), np.float32))
            last_prediction.append(np.zeros((2, 1), np.float32))
            current_prediction.append(np.zeros((2, 1), np.float32))
            kalman.setdefault(str(l), cv2.KalmanFilter(4, 2))
            kalman[str(l)].measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            kalman[str(l)].transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                                        [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
            kalman[str(l)].processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                                       [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
            # last_measurement = current_measurement = np.array((2, 1), np.float32)
            # last_prediction = current_prediction = np.zeros((2, 1), np.float32)
        # time.sleep(15)
        # i = 0

        # frame = np.zeros((960, 1280, 3), np.uint8)
        while True:
            print("!!!!!!!!!!!!!!!!!!!!", time.time())
            frame = np.zeros((960, 1280, 3), np.uint8)

            # if len(transList) != 0:
            #     x = int(transList[1][0])
            #     y = int(transList[1][1])
            #     self.targetMove(x, y, current_measurement[1], last_measurement[1],
            #                     current_prediction[1], last_prediction[1], kalman["1"], frame)

            for j in range(len(transList)):
                # x = int(transList[j][0])
                # y = int(transList[j][1])
                self.targetMove(int(transList[j][0]), int(transList[j][1]), current_measurement[j], last_measurement[j],
                                current_prediction[j], last_prediction[j], kalman[str(j)], frame)
                # self.targetMove(x, y, current_measurement[j], last_measurement[j],
                #                 current_prediction[j], last_prediction[j], kalman[str(j)], frame)

            cv2.imshow("kalman_tracker", frame)

            # print(transDict, transList, targetDict)
            print("@@@@@@@@@@@@@@@@@@@@", time.time())
            cv2.waitKey(1)
            # if (cv2.waitKey(1) & 0xff) == 27:
            #     break

    def contourTrack(self, transList):

        videoDir = "d:\\1\\Video_20200204122301684 .avi"
        avi = Video(videoDir)

        es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
        kernel = np.ones((5, 5), np.uint8)
        background = None

        while True:

            frame, nF, t = avi.getImageFromVideo()
            if background is None:
                background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                background = cv2.GaussianBlur(background, (21, 21), 0)
                continue  # 跳出这个循环
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

            diff = cv2.absdiff(background, gray_frame)
            diff = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)[1]
            diff = cv2.dilate(diff, es, iterations=2)
            image, cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            print(len(cnts))
            centerlist = []
            for c in cnts:

                if cv2.contourArea(c) < 1500:
                    continue

                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                transList.append([x + w / 2, y + h / 2])
                # transList = centerlist
            print(transList)

            cv2.imshow("contours", frame)  ##显示轮廓的图像
            cv2.imshow("dif", diff)

            if cv2.waitKey(1000 // 12) & 0xff == ord("q"):  ##改为//后就没有问题
                break

        cv2.destroyAllWindows()

    def contourTrackFromVision(self, transFrame):
        # time.sleep(15)
        while True:
            print("*" * 100)
            print(time.time())
            if transFrame is not None:

                print(np.frombuffer(transFrame, dtype=np.double).reshape((6, 7, 3)))

                # time.sleep(0.5)
                # cv2.imshow("contours", transFrame)
            cv2.waitKey(50)
            print(time.time())

    def read(self,transDict, transList, targetDict):
        time.sleep(0.001)
        print("!!!!!!!!!!!", )
        print(transList,)
        print("!!!!!!!!!!!", )


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
    #         tempDict, tempBottleDict, uuIDList, p0, label, centerLists =
    #         targetTracking.createTarget(dataDict, frame, featureimg,nFrame,_vision, _imgproc)
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
    trackFlag = 0

    with multiprocessing.Manager() as MG:  # 重命名

        transDict = MG.dict()
        transList = MG.list()
        targetDict = MG.dict()
        # transFrame = MG.Array("i", range(126))
        # transFrame = MG.Array("i", np.zeros((6, 7, 3), np.uint8))
        transFrame = multiprocessing.RawArray('d', np.zeros((6, 7, 3), np.double).ravel())
        # example rigion
        # transFrame = np.zeros((6, 7, 3), np.uint8)
        # cam, _image = imageInit()

        # p0 = multiprocessing.Process(target=track.contourTrack, args=(transList,))
        # p0.daemon = True
        # p0.start()
        # # p0.join()

        # p0 = multiprocessing.Process(target=track.contourTrackFromVision, args=(transFrame,))
        # p0.daemon = True
        # p0.start()

        ptest = multiprocessing.Process(target=track.read, args=(transDict, transList, targetDict))
        ptest.daemon = True
        ptest.start()

        # # first line code is without filter; second line one is with filter
        # p2 = multiprocessing.Process(target=track.trackProcess, args=(transDict, transList, targetDict))
        # # p2 = multiprocessing.Process(target=track.trackWithFilter, args=(transDict, transList, targetDict))
        # p2.daemon = True
        # p2.start()

        p1 = multiprocessing.Process(target=vision_run, args=(transDict, transList, targetDict, transFrame))
        p1.daemon = True
        p1.start()
        p1.join()
