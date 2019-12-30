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
        trackFlag = 0
        position = 0
        speed = 0
        angle = 0
        type = 0
        typeCounter = 0

        trackDict.setdefault("target", targetList)

        uuID = uuid.uuid1()
        targetList.append(uuID)
        targetList.append(trackFlag)
        targetList.append(position)
        targetList.append(speed)
        targetList.append(angle)
        targetList.append(type)
        targetList.append(typeCounter)

        print(trackDict)
        return trackDict

    def updateTarget(self,bottleDict):
        #将bottleDict中数据进行换算，并更新至trackDict内相对应的target
        bottleDict = self.bottleDict
        file = open("trackDict.txt", "r")
        trackDict = json.load(file.read())
        return trackDict

    #def insertTarget(self):

if __name__ == "__main__":
    track = Track().createTarget()