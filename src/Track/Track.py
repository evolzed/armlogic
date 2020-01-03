# -- coding: utf-8 --
# !/bin/python
import os
import sys
import uuid


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

        # 创建新的target并标记uuid
        targetDict = dict()
        targetLists = list()
        trackFlag = "0"
        position = "0"
        speed = "0"
        angle = "0"
        type = "0"
        typeCounter = "0"
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
        print(targetDict)
        return targetDict

    def updateTarget(self,targetDict):
        """
        更新target功能，自定义时间间隔Δt = （t2 - t1），主流程会根据该时间间隔进行call bgLearn；

        :param targetDict: 上一步的目标物的信息
        :return: 同一UUID下的目标物的信息更新；
        """
        deltaT = 0.02

        oldTargetDict = targetDict
        newTargetDict = oldTargetDict
        timeCost = newTargetDict.get("timeCost")
        newTargetDictLists = oldTargetDict.get("target")
        # 循环遍历，更新target
        for i in range(len(newTargetDictLists)):
            newTargetDictLists[i][2] = str(float(newTargetDictLists[i][3]) * deltaT)

        newTargetDict.setdefault("targetTrackTime", str(float(timeCost) + deltaT) )
        # print(newTargetDict)
        # print(float(newTargetDictLists[1][3]) * deltaT)
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

    # 测试用例，此处bottleDict使用的非BS0.1中bottledict，而是将来为Main中提供的传参！
    bottledict1 = {'target': ["da5b6600-2b6e-11ea-8937-985fd3d62bfb", "11", "66", "11", "11", "11", "11"],
                   'bgTimeCost': 0.09634879999999946, 'timeCost': 1578021152.9692435, 'nFrame': 222}
    bottledict2 = {'target': [["f025d3fe-2b6e-11ea-a086-985fd3d62bfb", "11", "77", "11", "11", "11", "11"],
                              ["kkkkkkkk-2b6e-11ea-a086-985fd3d62bfb", "11", "88", "11", "11", "11", "11"]],
                   'bgTimeCost': 0.10440749999999888, 'timeCost': 1578021153.380255, 'nFrame': 229}
    # track1 = Track().createTarget(bottledict1)        # 测试createTarget
    track2 = Track().updateTarget(bottledict2)      # 测试updateTarget
