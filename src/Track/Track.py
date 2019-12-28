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
        return trackDict

    def updateTarget(self,bottleDict):
        #将bottleDict中数据进行换算，并更新至trackDict内相对应的target
        file = open("trackDict.txt", "r")
        trackDict = json.load(file.read())
        return trackDict
