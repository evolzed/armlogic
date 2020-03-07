# -- coding: utf-8 --
# !/bin/python
import math
import os
import sys
import cv2
import uuid
from src.BS02.camera import Camera

from src.BS02.vision import *
from src.BS02.imageProcess.imgProc import *
import time
from lib.Sort.sort import Sort
from timeit import default_timer as timer
import numpy as np
from operator import itemgetter
from src.BS02.yolo.Yolo import YOLO
from multiprocessing import Process
import multiprocessing
from lib.Logger.Logger import Logger

"""
from Track_duplication, start from 20200203
"""
# sys.stdout = Logger("d:\\12.txt")  # 保存到D盘

speedMonitorList = [[]]

preVList = [[0 for i in range(5)] for j in range(20)]
preV = dict()
preV.setdefault("preVList", preVList)

cnt = 0
sort = Sort()
tempT = preV["t"] = time.time()

# ###the initial time of Flag(call detectObj)###
timeOfFlag = time.time()
countOfTime = 0

# define the checkTime
checkTime = 0


class Track:
    """
    提供更新实时Target目标功能，updateTarget()；
    提供增加新的Target目标功能，createTarget()；
    提供合并target目标功能，mergeTarget()；
    提供检查Target目标功能，checkTarget()实现实时运行中为实际镜头图像范围内的所有目标物；
    根据前一帧与当前帧的target信息，对target的速度进行估计计算(待开发)；
    提供总进程trackProcess()；
    """

    def createTarget(self, transDict, transList , Flag):
        """
        提供增加新的Target目标功能，createTarget()；
        :param transDict: 由vision输出的bottleDict的copy
        :param transList: 由vision输出的LKtrackedList的copy
        :param Flag: 调用detectObj机制
        :return: targetDict
        """

        startTime = time.time()
        deltaT = 0.01
        global trackFlag
        targetDict = dict()
        speed = [0, 0]
        # speed = [24.83, 0]  # 设置初始速度，为录像视频中平均速度; 初始像素速度为5.8， 转换成时间相关速度 近似值#
        angle = [0, 0]
        type = 0
        typCounter = 0
        time.sleep(0.001)
        tempList = [[] for j in range(len(transList))]

        if len(transList) != 0:

            for i in range(len(transList)):

                tempList[i].append(str(uuid.uuid1()))  # 对应位置打上uuID
                # tempList[i].append(str(i) + "**********")    # 测试用

                if Flag is not None and len(Flag) != 0:
                    tempList[i].append(Flag[0])

                tempList[i].append([transList[i][0], transList[i][1]])
                tempList[i].append(speed)
                tempList[i].append(angle)
                tempList[i].append(type)
                tempList[i].append(typCounter)

            time.sleep(deltaT - 0.0025)  # 实际让程序运行总体控制在0.01s内；

        targetDict.setdefault("target", tempList)

        # 增加timeCost  和  targetTrackTime。。。
        timeCost = time.time()
        targetTrackTime = startTime + deltaT
        targetDict.setdefault("timeCost", timeCost)
        targetDict.setdefault("targetTrackTime", targetTrackTime)

        return targetDict

    def updateTarget(self, targetDict, Flag):
        """
        更新target功能，自定义时间间隔Δt = （t2 - t1），主流程会根据该时间间隔进行
        :param targetDict: 上一步的目标物的信息
        :param Flag: 调用detectObj机制，目前嵌入在本方法内，后续可开放至总进程！；
        :return: 同一UUID下的目标物的信息更新,新的targetDict
        """

        global timeOfFlag, countOfTime, checkTime
        startTime = time.time()
        deltaT = 0.01  # 可调时间步长
        tempTargetDict = targetDict  # updateTarget()进程中，每次存储的临时TargetDict # ；

        # 每步updateTarget()进行自主估值
        if len(tempTargetDict["target"]) != 0:

            # # SpeedMonitor in trackProcess, may be develop in the future
            # if True:
            #     # speedMonitor,先时时刻刻监测，
            #     self.speedMonitor(transDict, transList, tempTargetDict)

            for i in range(len(tempTargetDict["target"])):
                tempTargetDict["target"][i][2][0] =\
                    tempTargetDict["target"][i][2][0] + tempTargetDict["target"][i][3][0] * (deltaT + checkTime)

                tempTargetDict["target"][i][2][1] =\
                    tempTargetDict["target"][i][2][1] + tempTargetDict["target"][i][3][1] * (deltaT + checkTime)

                # Flag[]
                tempTargetDict["target"][i][1] = Flag[0]

                # 回传targetDict
        targetDict = tempTargetDict

        # targetDict.update(tempTargetDict)    # 更新成targetDict
        time.sleep(deltaT - 0.005)  # 实际让程序运行总体控制在0.01s内；
        timeCost = time.time()
        targetTrackTime = startTime + deltaT
        targetDict["timeCost"] = timeCost
        targetDict["targetTrackTime"] = targetTrackTime

        # ###the function of setting Flag to call detectObj()###
        if time.time() - timeOfFlag > countOfTime * 3:

            Flag[0] = 0
            countOfTime += 1

        checkTime = 0
        
        return targetDict

    def mergeTarget(self, targetDictNew, targetDictHistory):
        """
        合并target功能，将新target与原先target合并；实现实时运行中为实际镜头图像范围内的所有目标物
        :param targetDictNew: 比较的新创立的target
        :param targetDictHistory: 比较的原先的在运行过程中的target
        :return: 合并后的target
        """

        tempList = targetDictHistory.get("target")

        for i in range(len(targetDictNew.get("target"))):

            tempList.append(targetDictNew.get("target")[i])

        targetDictHistory.setdefault("target", tempList)

        return targetDictHistory

    def checkTarget(self, transDict, targetDict, transList, Flag):
        """
        检查target功能，在trackProcess循环末期，对照自身target信息和vision的信息；
        :param transDict: 由vision输出的bottleDict的copy
        :param targetDict: 上一步的目标物的信息
        :param transList: 由vision输出的LKtrackedList的copy
        :param Flag: 调用detectObj机制
        :return: 修正的targetDict
        """

        targetFound = 0
        deltaDistance = 190
        global checkTime
        startTime = time.time()
        
        # 目前 按照顺序依次对targetDict 中的list 与 transList 进行比较， 目前，对于循环末期，直接赋值transList
        # compare机制，用临时存储tempList，比较其长度与targetDict长度，假如追踪target在判断范围内，则不做修正，
        tempTargetDict = targetDict
        tempList = sorted(transList.__deepcopy__({}), key=itemgetter(0, 1), reverse=True)
        tempListCopy = tempList.copy()

        # 遍历比较：
        if len(tempList) != 0 and len(tempTargetDict) != 0:

            for ii in range(len(tempList), 0, -1):

                for jj in range(len(tempTargetDict["target"])):

                    if math.sqrt(pow((tempTargetDict["target"][jj][2][0] - tempList[ii - 1][0]), 2) +
                                 pow((tempTargetDict["target"][jj][2][1] - tempList[ii - 1][1]), 2)) < deltaDistance:
                        # 意味着监测到的数据小范围周边已经有至少一个在追踪的target；认为不必要添加新的target；首先只赋予速度（历史速度）：
                        # 假设有速度值 (沿着皮带方向)：
                        if len(tempList[ii - 1]) > 5 and tempList[ii - 1][5] > 0:

                            tempTargetDict["target"][jj][3][0] = tempList[ii - 1][5]
                            tempTargetDict["target"][jj][3][1] = tempList[ii - 1][6]
                        targetFound += 1

                if targetFound > 0:

                    tempListCopy.pop(ii - 1)
                    targetFound = 0

            tempTargetDict = self.mergeTarget(tempTargetDict, self.createTarget(transDict, tempListCopy, Flag))

        targetDict.update(tempTargetDict)

        checkTime = time.time() - startTime
        
        return targetDict

    def trackProcess(self, transDict, transList, targetDict, Flag):
        """
        track总进程，自身根据速度与位移间关系更新target信息，固定计数为一个周期，周期末进行检查，对反应现实中的target信息和视觉输出信息作比较；
        :param transDict: 由vision输出的bottleDict的copy
        :param transList: 由vision输出的LKtrackedList的copy
        :param targetDict: 目标物的信息
        :param Flag: 调用detectObj机制
        :return: None
        """

        while True:

            trackFrame = np.zeros((960, 1280, 3), np.uint8)

            if len(transList) != 0:
                
                for i in range(10):
                    
                    # 更新target；
                    if i < 9:
                        
                        if len(targetDict) == 0:

                            newTargets = self.createTarget(transDict, transList, Flag)
                            targetDict = self.updateTarget(newTargets, Flag)
                            
                        else:
                            
                            targetDict = self.updateTarget(targetDict, Flag)
                            
                        print("targetDict!!!", targetDict)

                    if i == 9:

                        # 更新时，要求在targetTrackTime上做自加，在最后一次子循环 中 作判断
                        # # 目前 按照顺序依次对targetDict 中的list 与 transList 进行比较；
                        targetDict = self.checkTarget(transDict, targetDict, transList, Flag)

            else:

                time.sleep(0.001)

            # show
            if len(transList) != 0 and len(targetDict) != 0:
                
                tempList = sorted(transList.__deepcopy__({}), key=itemgetter(0, 1), reverse=True)
                
                # vision_Run show
                for ii in range(len(tempList)):
                    
                    visionX, visionY = tempList[ii][0], tempList[ii][1]
                    cv2.circle(trackFrame, (visionX, visionY), 12, (0, 100, 0), 3)

                for jj in range(len(targetDict["target"])):
                    
                    currentX, currentY = int(targetDict["target"][jj][2][0]), int(targetDict["target"][jj][2][1])
                    
                    # FilterShow
                    # lastX, lastY = int(lastDict["target"][jj][2][0]), int(lastDict["target"][jj][2][1],)
                    uuIDText = targetDict["target"][jj][0]
                    cv2.circle(trackFrame, (currentX, currentY), 6, (0, 0, 200), 2)
                    cv2.putText(trackFrame, uuIDText, (currentX - 100, currentY + 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 2, cv2.LINE_AA)
                    
            cv2.imshow("target", trackFrame)
            
            if (cv2.waitKey(30) & 0xff) == 27:
                break


if __name__ == "__main__":

    track = Track()

    with multiprocessing.Manager() as MG:  # 重命名

        transDict = MG.dict()
        transList = MG.list()
        targetDict = MG.dict()

        transFrame = multiprocessing.RawArray('d', np.zeros((6, 7, 3), np.double).ravel())
        # example rigion

        ptest = multiprocessing.Process(target=track.read, args=(transDict, transList, targetDict))
        ptest.daemon = True
        ptest.start()

        p1 = multiprocessing.Process(target=vision_run, args=(transDict, transList, targetDict, transFrame))
        p1.daemon = True
        p1.start()
        p1.join()
