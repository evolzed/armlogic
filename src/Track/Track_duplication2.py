# -- coding: utf-8 --
# !/bin/python
import os
import sys
import numpy as np
import uuid
from src.Vision.camera import Camera
import cv2
from src.Vision.imageProcess.imgProc_duplication import ImgProc
import time
from src.Vision.yolo.Yolo import *
from timeit import default_timer as timer
from src.Vision.vision_duplication import *

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
        frameTime = bottleDict.get("frameTime")
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
        targetDict.setdefault("frameTime", frameTime)


        # tempList.append('\n')

        # file = open("targetDict_test.txt", "a")
        # for target in tempList:
        #     file.writelines(target + ", ")
        # file.writelines("\n")
        # print(targetDict, uuIDList)
        return targetDict, uuIDList

    def updateTarget(self, targetDict, _frame):
        """
        更新target功能，自定义时间间隔Δt = （t2 - t1），主流程会根据该时间间隔进行call bgLearn；

        :param targetDict: 上一步的目标物的信息
        :return: 同一UUID下的目标物的信息更新；
        """
        self._frame = _frame
        deltaT = 0.01

        oldTargetDict = targetDict
        newTargetDict = oldTargetDict
        frameTime = newTargetDict.get("frameTime")
        newTargetDictLists = oldTargetDict.get("target")
        # 循环遍历，更新target，
        for i in range(len(newTargetDictLists)):
            newTargetDictLists[i][2][0] = newTargetDictLists[i][2][0] + float(newTargetDictLists[i][3][0]) * (deltaT)
            newTargetDictLists[i][2][1] = newTargetDictLists[i][2][1] + float(newTargetDictLists[i][3][1]) * (deltaT)
            cv2.rectangle(_frame, (int(newTargetDictLists[i][2][0] - 100), int(newTargetDictLists[i][2][1] - 100)),
                          (int(newTargetDictLists[i][2][0]) + 100, int(newTargetDictLists[i][2][1]) + 100), (125, 0, 125), 4)
            # print(i)
        # targetTrackTime 更新为Δt后：
        newTargetDict["targetTrackTime"] = frameTime + (deltaT)
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

    cam, _image = imageInit()

    yolo = YOLO()
    _vision = Vision(cam, yolo, imgproc_=None)
    _imgproc = ImgProc(10)
    prev_time = timer()
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    # if sysArc[0] == "Linux" and sysArc[-1] == "aarch64":
    #     print("press_any_key_exit!")
    #     cam.press_any_key_exit()

    # trackObj = ImageTrack()
    preframe, nFrame, t = cam.getImage()
    preframeb, bgMaskb, resarrayb = _imgproc.delBg(preframe) if _imgproc else (preframe, None)
    k = 1
    startt = timer()
    left = 0
    top = 0
    right = 0
    bottom = 0
    flag = 0
    inputCorner = np.array([])

    feature_params = dict(maxCorners=30,
                          qualityLevel=0.3,
                          minDistance=7,  # min distance between corners
                          blockSize=7)  # winsize of corner
    # params for lk track
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p0 = np.array([])
    label = np.array([])

    _frame, nFrame, t = cam.getImage()
    frame, bgMask, resarray = _imgproc.delBg(_frame) if _imgproc else (_frame, None)
    # cv2.namedWindow("kk", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("kk", frame)
    # cv2.waitKey(3000)
    # global prev_time
    # 设定计时器, 统计识别图像耗时
    # prev_time = timer()
    # 将opencv格式的图像数据转换成PIL类型的image对象，便于进行标框和识别效果可视化
    img = PImage.fromarray(frame)  # PImage: from PIL import Vision as PImage
    # img.show()
    # feed data into model
    dataDict = _vision.yolo.detectImage(img)
    dataDict["bgTimeCost"] = _imgproc.bgTimeCost if _imgproc else 0
    result = np.asarray(dataDict["image"])
    # dataDict["image"] = result  # result：cv2.array的图像数据
    dataDict["image"] = img  # img：Image对象
    # dataDict["timeCost"] = exec_time
    dataDict["nFrame"] = nFrame
    dataDict["frameTime"] = t  # 相机当前获取打当前帧nFrame的时间t
    print(dataDict)
    dataDict["box"] = []
    tempDict, uuID = Track().createTarget(dataDict)

    while True:
        _frame, nFrame, t = cam.getImage()
        camfps = " Cam" + cam.getCamFps(nFrame)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "NetFPS:" + str(curr_fps)
            curr_fps = 0

        # frame, bgMask, resarray = _imgproc.delBg(_frame) if _imgproc else (_frame, None)
        # # cv2.namedWindow("kk", cv2.WINDOW_AUTOSIZE)
        # # cv2.imshow("kk", frame)
        # # cv2.waitKey(3000)
        # # global prev_time
        # # 设定计时器, 统计识别图像耗时
        # # prev_time = timer()
        # # 将opencv格式的图像数据转换成PIL类型的image对象，便于进行标框和识别效果可视化
        # img = PImage.fromarray(frame)  # PImage: from PIL import Vision as PImage
        # # img.show()
        # # feed data into model


        dataDict["bgTimeCost"] = _imgproc.bgTimeCost if _imgproc else 0
        result = np.asarray(dataDict["image"])
        # dataDict["image"] = result  # result：cv2.array的图像数据
        dataDict["image"] = img  # img：Image对象
        # dataDict["timeCost"] = exec_time
        dataDict["nFrame"] = nFrame
        dataDict["frameTime"] = t  # 相机当前获取打当前帧nFrame的时间t

        dataDict = _vision.yolo.detectImage(img)
        print(tempDict)
        if tempDict["target"] is None:
            tempDict = Track().updateTarget(tempDict, _frame)
        else:
            tempDict2, uuID2 = Track().createTarget(dataDict)
            Track().mergeTarget(tempDict2, tempDict)
            tempDict = Track().updateTarget(tempDict, _frame)

        # arr = np.asarray(dataDict["image"])
        imglist = _imgproc.getBoxOnlyPic(dataDict, preframe)
        imglistk = _imgproc.getBoxOnlyPic(dataDict, _frame)
        drawimg = frame.copy()
        featureimg = cv2.cvtColor(preframeb, cv2.COLOR_BGR2GRAY)
        secondimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # tempT = None
        # if tempDict.get("frameTime") is not None:
        #     if tempT is None:
        #         tempT = 0
        #     tempT = tempT + t - tempDict.get("frameTime")
        #     # print(str(tempDict["frameTime"]) + ",   " + str(t) + ",   " + str(tempT))
        #     if tempT > 10:
        #         tempT = 0
        #         tempDict3, uuID2 = Track().createTarget(dataDict)


        # detect
        if flag == 0:
            p0, label = _imgproc.detectObj(featureimg, drawimg, dataDict, feature_params, 3)
            if p0 is not None and label is not None:

                flag = 1
        # track
        else:
            p0, label = _imgproc.trackObj(featureimg, secondimg, drawimg, label, p0, lk_params)

            # tempDict = Track().updateTarget(tempDict, drawimg)

        # clear
        if "box" not in dataDict:
            p0 = np.array([])
            label = np.array([])
            flag = 0
            cv2.circle(drawimg, (100, 100), 15, (0, 0, 255), -1)  # red  track
        # tempDict = Track().updateTarget(tempDict, _frame)

        cv2.imshow("res", _frame)
        cv2.waitKey(10)
        preframeb = frame.copy()

        if bgMask is not None:
            dataDict = _imgproc.getBottlePose(_frame, bgMask, dataDict)
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.putText(result, text=camfps, org=(150, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(0, 255, 255), thickness=2)
        cv2.imshow("result", result)
        # cv2.waitKey(1000)
        cv2.waitKey(10)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # return dataDict
        global bottleDict
        bottleDict = dataDict
    cam.destroy()