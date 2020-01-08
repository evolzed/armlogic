# -- coding: utf-8 --
# !/bin/python
import os
import sys
import uuid
from src.Vision.camera import Camera
import cv2
from src.Vision.imageProcess.imgProc import ImgProc
import time

class Track:
    """
    提供增加新的Target目标功能
    提供更新实时Target目标功能
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
        # tempList.append('\n')

        # file = open("targetDict_test.txt", "a")
        # for target in tempList:
        #     file.writelines(target + ", ")
        # file.writelines("\n")
        # print(targetDict, uuIDList)
        return targetDict, uuIDList

    def updateTarget(self, targetDict):
        """
        更新target功能，自定义时间间隔Δt = （t2 - t1），主流程会根据该时间间隔进行call bgLearn；

        :param targetDict: 上一步的目标物的信息
        :return: 同一UUID下的目标物的信息更新；
        """
        deltaT = 0.1

        oldTargetDict = targetDict
        newTargetDict = oldTargetDict
        frameTime = newTargetDict.get("frameTime")
        newTargetDictLists = oldTargetDict.get("target")
        # 循环遍历，更新target，
        for i in range(len(newTargetDictLists)):
            newTargetDictLists[i][2][0] = newTargetDictLists[i][2][0] + float(newTargetDictLists[i][3][0]) * (10 * deltaT)
            newTargetDictLists[i][2][1] = newTargetDictLists[i][2][1] + float(newTargetDictLists[i][3][1]) * (10 * deltaT)
            cv2.rectangle(_frame, (int(newTargetDictLists[i][2][0]), int(newTargetDictLists[i][2][1])),
                          (int(newTargetDictLists[i][2][0]) + 50, int(newTargetDictLists[i][2][0]) + 50), (125, 0, 125), 4)
            # print(i)
        # targetTrackTime 更新为10倍Δt后：
        newTargetDict["targetTrackTime"] = frameTime + (10 * deltaT)
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

    # # 测试用例，此处bottleDict使用的非BS0.1中bottledict，而是将来为Main中提供的传参！
    # bottledict1 = {'target': ["da5b6600-2b6e-11ea-8937-985fd3d62bfb", 0, [300, 300], [10, 10], 0, 0, 0],
    #                'bgTimeCost': 0.09634879999999946, 'timeCost': 1578021152.9692435, 'nFrame': 222, 'frameTime': 0}
    # bottledict2 = {'target': [["f025d3fe-2b6e-11ea-a086-985fd3d62bfb", 0, [300, 300], [10, 10], 0, 0, 0],
    #                           ["kkkkkkkk-2b6e-11ea-a086-985fd3d62bfb", 0, [400, 400], [10, 10], 0, 0, 0]],
    #                'bgTimeCost': 0.10440749999999888, 'timeCost': 1578021153.380255, 'nFrame': 229, 'frameTime': 0}
    # # track1 = Track().createTarget(bottledict1)        # 测试createTarget
    # track2 = Track().updateTarget(bottledict2)      # 测试updateTarget

    cam = Camera()

    bottleDict = {"image": 0, "box": [(3, 0.9, 0, 0, 200, 200),
                                      (2, 0.9, 0, 0, 500, 500)],
                  "bgTimeCost": 0, "timeCost": 0, "nFrame": 0}

    targetDict = {"target": [["f025d3fe-2b6e-11ea-a086-985fd3d62bfb", 0, [100, 100], [50, 50], 0, 0, 0],
                              ["kkkkkkkk-2b6e-11ea-a086-985fd3d62bfb", 0, [400, 400], [50, 50], 0, 0, 0]],
                  "bgTimeCost": 0.10440749999999888, "timeCost": 1578021153.380255, "Frame": 0, "frameTime": 0, "targetTrackTime":0}
    # tempDict = targetDict

    # tempDict2, uuID = Track().createTarget(bottleDict)

    # tempDict3 = Track().mergeTarget(tempDict, tempDict2)

    tempDict, uuID = Track().createTarget(bottleDict)

    while True:

        _frame, nFrame, t = cam.getImage()
        tempDict["nFrame"] = nFrame
        tempT = None

        # 虚拟间隔时间增加targetDict，实际后续由
        if tempDict.get("frameTime") is not None:
            print(str(tempDict["frameTime"]) + ",   " + str(t))
            if tempT is None or tempT == 0 :
                tempT =0
                tempT = tempT + t - tempDict.get("frameTime")
                print(tempT)
                if tempT > 2:
                    tempT = 0
                    tempDict3, uuID2 = Track().createTarget(bottleDict)
                    Track().mergeTarget(tempDict3, tempDict)

        tempDict["frameTime"] = t

        # 判断条件
        if (tempDict["targetTrackTime"] == 0 or abs(t - tempDict["targetTrackTime"]) < 0.08 ):
            tempDict = Track().updateTarget(tempDict)

        cv2.imshow("test", _frame)
        tempImgproc = ImgProc(10)

        frame, bgMask, resarray = tempImgproc.delBg(_frame) if tempImgproc else (_frame, None)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break