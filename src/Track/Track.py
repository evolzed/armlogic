# -- coding: utf-8 --
# !/bin/python
import os
import sys
import json
import uuid

class Track:

    def createTarget(self):
        #创建新的target并标记uuid
        trackDict = dict()
        targetList = list()
        trackFlag = "0"
        position = "0"
        speed = "0"
        angle = "0"
        type = "0"
        typeCounter = "0"

        trackDict.setdefault("target", targetList)

        uuID = str(uuid.uuid1())
        targetList.append(uuID)
        targetList.append(trackFlag)
        targetList.append(position)
        targetList.append(speed)
        targetList.append(angle)
        targetList.append(type)
        targetList.append(typeCounter)
        #targetList.append('\n')

        file = open("trackDict_test.txt", "a")
        for target in targetList:
            file.writelines(target + ", ")
        file.writelines("\n")
        print(trackDict)
        return trackDict

    def updateTarget(self,bottleDict):
        #将bottleDict中数据进行换算，并更新至trackDict内相对应的target
        #bottleDict = self.bottleDict
        file = open("trackDict_test.txt", "a")

        trackDict = json.load(file.read())
        print(trackDict)
        return trackDict


if __name__ == "__main__":
    #os.mknod("trackDict.txt")

    bottledict1 = {'target': ["2add89f6-2aa8-11ea-921f-985fd3d62bfb", "11", "11", "11", "11", "11", "11"]}
    track1 = Track().createTarget()
    #track2 = Track().updateTarget(bottledict1)
