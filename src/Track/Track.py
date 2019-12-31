# -- coding: utf-8 --
# !/bin/python
import os
import sys
import json
import uuid

#提供增加新的Target目标功能
#提供更新实时Target目标功能

class Track:

    def createTarget(self):

        """
        增加新的Target目标功能

        Returns: 新的带UUID的trackDict

        """

        #创建新的target并标记uuid
        trackDict = dict()
        targetLists = list()
        trackFlag = "0"
        position = "0"
        speed = "0"
        angle = "0"
        type = "0"
        typeCounter = "0"

        trackDict.setdefault("target", targetLists)

        uuID = str(uuid.uuid1())    #自己创建，后期使用LKTrack打上的UUID
        targetLists.append(uuID)
        targetLists.append(trackFlag)
        targetLists.append(position)
        targetLists.append(speed)
        targetLists.append(angle)
        targetLists.append(type)
        targetLists.append(typeCounter)
        #targetLists.append('\n')

        file = open("trackDict_test.txt", "a")
        for target in targetLists:
            file.writelines(target + ", ")
        file.writelines("\n")
        print(trackDict)
        return trackDict

    def updateTarget(self,bottleDict):
        """
        更新实时target功能

        Args:
            bottleDict: 经过计算的LKTrack 的输出

        Returns: 同一UUID的信息更新；
        """
        #将bottleDict中数据进行换算，并更新至trackDict内相对应的target
        #bottleDict = self.bottleDict
        file = open("trackDict_test.txt", "r+")

        # 逐行读取多行文件中的trackDict，与更新成UUID为相同一个的bottleDict中的值
        while True:
            targetLists = file.readlines(10000)

            if not targetLists:
                break
            for targetList in targetLists:
                # 对比UUID ，假如一样则执行更新
                #临时tempLists
                tempLists = bottleDict.get("target")
                print(tempLists)
                tempSingleList = targetList.split(", ")
                #print(tempLists)
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
                #print(targetList)
        file.close()


if __name__ == "__main__":

    #测试用例，此处bottleDict使用的非BS0.1中bottledict，而是将来为Main中提供的传参！
    bottledict1 = {'target': ["da5b6600-2b6e-11ea-8937-985fd3d62bfb", "11", "66", "11", "11", "11", "11"]}
    bottledict2 = {'target': ["f025d3fe-2b6e-11ea-a086-985fd3d62bfb", "11", "77", "11", "11", "11", "11"]}
    #track1 = Track().createTarget()        #测试createTarget
    track2 = Track().updateTarget(bottledict2)      #测试updateTarget
