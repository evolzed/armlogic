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

    def createTarget(self,bottleDict):
        """
        增加新的Target目标功能

        :return: 新的带UUID的targetDict
        """

        # 创建新的target并标记uuid 返回给bottleDict
        targetDict = dict()
        targetLists = list()
        trackFlag = 0
        position = [int((bottleDict["box"][0][2] + bottleDict["box"][0][4]) / 2),
                    int((bottleDict["box"][0][3] + bottleDict["box"][0][5]) / 2)]
        speed = [10, 10]
        angle = 0
        type = 0
        typeCounter = 0
        nFrame = bottleDict.get("nFrame")
        bgTimeCost = bottleDict.get("bgTimeCost")
        timeCost = bottleDict.get("timeCost")

        targetDict.setdefault("target", targetLists)

        uuID = str(uuid.uuid1())    # 自己创建，用uuid1 打上 UUID
        targetLists.append(uuID)
        targetLists.append(trackFlag)
        targetLists.append(position)
        targetLists.append(speed)
        targetLists.append(angle)
        targetLists.append(type)
        targetLists.append(typeCounter)
        targetDict.setdefault("nFrame", nFrame)
        targetDict.setdefault("bgTimeCost", bgTimeCost)
        targetDict.setdefault("timeCost", timeCost)
        # targetLists.append('\n')

        # file = open("targetDict_test.txt", "a")
        # for target in targetLists:
        #     file.writelines(target + ", ")
        # file.writelines("\n")
        print(targetDict, uuID)
        return targetDict, uuID

    def updateTarget(self,targetDict):
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
        print("frameTime:" + str(newTargetDict["frameTime"]) + "     targetTrackTime:" + str(newTargetDict["targetTrackTime"])  + "     realTime:" + str(time.time()))
        return newTargetDict

    def checkTarget(self,bottleDict):
        """
        检查target功能，自定义时间间隔Δt = （t2 - t1），主流程会根据该时间间隔进行call bgLearn；

        :param bottleDict: 上一步的
        :return: 同一UUID的信息更新；
        """

        # 将bottleDict中数据进行换算，并更新至targetDict内相对应的target

        file = open("targetDict_test.txt", "r+")

        # 逐行读取多行文件中的targetDict，与更新成UUID为相同一个的bottleDict中的值
        while True:
            targetLists = file.readlines(10000)

            if not targetLists:
                break
            for targetList in targetLists:
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

    targetDict = {'target': [["f025d3fe-2b6e-11ea-a086-985fd3d62bfb", 0, [100, 100], [50, 50], 0, 0, 0],
                              ["kkkkkkkk-2b6e-11ea-a086-985fd3d62bfb", 0, [400, 400], [50, 50], 0, 0, 0]],
                  'bgTimeCost': 0.10440749999999888, 'timeCost': 1578021153.380255, 'nFrame': 0, 'frameTime': 0, 'targetTrackTime':0}
    tempdict = targetDict

    Track().createTarget(bottleDict)

    while True:
        _frame, nFrame, t = cam.getImage()
        tempdict["nFrame"] = nFrame
        tempdict["frameTime"] = t

        if (tempdict["targetTrackTime"] == 0 or abs(t - tempdict["targetTrackTime"]) < 0.08 ):
            tempdict = Track().updateTarget(tempdict)
        cv2.imshow("test", _frame)
        tempImgproc = ImgProc(10)

        frame, bgMask, resarray = tempImgproc.delBg(_frame) if tempImgproc else (_frame, None)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break