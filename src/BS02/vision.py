import os
import sys

from src.BS02.track.track import *
from src.BS02.imageProcess.imageTools import eDistance
from tools.costTimeCal import CostTimeCal
sys.path.append(os.path.abspath("../../"))
import numpy as np
from ctypes import *
from timeit import default_timer as timer
import cv2
from src.BS02.imageProcess.imgProc import *
from src.BS02.video import Video
from src.BS02.interface import *
import time
import multiprocessing
# sys.path.insert(0, os.path.split(__file__)[0])
# from lib.GrabVideo import GrabVideo
import platform
import copy
from lib.Logger.Logger import Logger

# sys.stdout = Logger("d:\\12.txt")  # 保存到D盘

import matplotlib.pyplot as plt

sysArc = platform.uname()
if sysArc[0] == "Windows":
    from lib.HikMvImport_Win.utils.CameraParams_header import MV_FRAME_OUT_INFO_EX
elif sysArc[0] == "Linux" and sysArc[-1] == "aarch64":
    from lib.HikMvImport_TX2.utils.CameraParams_header import MV_FRAME_OUT_INFO_EX
else:
    print("不支持的系统架构，仅支持win10_64 和 Ubuntu16.04 ARM aarch64！！")
    sys.exit()
from src.BS02.camera import Camera, g_bExit
from src.BS02.yolo.Yolo import *

# from src.Vision.imageProcess.bgLearn import Bglearn
# from src.Vision.imageProcess.imageTrack import ImageTrack
gState = 1
bottleDict = {
    "image": None,
    # "box": ["predicted_class", "score", "left", "top", "right", "bottom", "angle", "diameter"],
    "box": None,
    "timeCost": None,
    "bgTimeCost": None,
    "nFrame": None,
    "frameTime": None,
    "getPosTimeCost": None,
    "isObj": False  # bool
}

#dataDict box 形式
# boxList.append([predicted_class, score, left, top, right, bottom, \
#                 angle, diameter, centerX, centerY, trackID, deltaX, deltaY, speedX, speedY])

#保持False 就可以
crop = False
#切换使用相机还是视频
useCamra = True

statisticTrackTime =True

videoDir = "E:\\1\\两个瓶子.avi"
bgDir = "E:\\1\\一个瓶子背景.avi"

LkTrackTimeOBJ = CostTimeCal("LkTrackTime", False)
YoloTimeOBJ = CostTimeCal("YoloTime", False)
BgLearnTimeOBJ = CostTimeCal("BgLearnTime", False)
DetectTimeOBJ = CostTimeCal("DetectTime", True)
state = False
# videoDir = "d:\\1\\Video_20200204122733339.mpone4"
# bgDir = "d:\\1\\背景1.avi"
gk = 0 #用于计数

drawimg = np.array([])
class Vision(object):
    """create main Vision class for processing images"""

    def __init__(self, imgCapObj, yolo, imgproc_=None):
        """相机自检"""
        self.cam = imgCapObj
        self.yolo = yolo
        self.imgproc = imgproc_
        # self.deviceNum = cam.getDeviceNum()
        # cam._data_buf, cam._nPayloadsize = self.cam.connectCam()
        # if self.imgproc.imgCap.cam is not None:
        #     if -1 == self.cam._data_buf:
        #         print("相机初始化失败！退出程序！")
        #         sys.exit()
        print("相机或视频初始化完成！")

    # 把新数据融合到tracked中
    def dataFusion(self, toBeTrack, tracked,label):
        global gk
        global drawimg
        ak =1
        toBeTrackID = []
        trackedID = []
        for elem in toBeTrack:
            toBeTrackID.append(elem[2])
        for elemTracked in tracked:
            trackedID.append(elemTracked[2])
        print("toBeTrackID::::", toBeTrackID)
        print("trackedID::::", trackedID)

        lenth = len(tracked)
        maxNo = max(trackedID)
        for elemIndex in range(len(toBeTrack)):  #遍历检测的中心点
            point = np.array([toBeTrack[elemIndex][0], toBeTrack[elemIndex][1] ])
            # distantList = []
            distantList = [0 for x in range(0, len(tracked))]
            for elemTrackedindex in range(len(tracked)):  #遍历跟踪的中心点
                pointTracked = np.array([tracked[elemTrackedindex][0], tracked[elemTrackedindex][1]])
                distant = eDistance(pointTracked, point)
                print("distant_between" + str(elemTrackedindex) + "and" + str(elemIndex)+".....", distant)
                distantList[elemTrackedindex] = distant
                #找出跟踪的中心点中 最接近检测中心点的一个中心点，把它的ID 和检测中心点的ID融合起来
            mini = min(distantList)
            print("distantList", distantList)
            if mini < 150:
                neareassIndex = distantList.index(min(distantList))
                toBeTrack[elemIndex][2] = tracked[neareassIndex][2]
                # print("toBeTrack[elemIndex][3]", toBeTrack[elemIndex][3])
                label[toBeTrack[elemIndex][3]] = tracked[neareassIndex][2]
                # print("label:", label)
                print("find the nearest!!!!!!fusion,%f===== %f" %(toBeTrack[elemIndex][2], tracked[neareassIndex][2]))
            else:
                toBeTrack[elemIndex][2] = maxNo + ak
                label[toBeTrack[elemIndex][3]] = maxNo + ak
                print("> 150!!!!!ADD CHANGE,%f ===== %f" %(toBeTrack[elemIndex][2], maxNo + ak))
                ak += 1

            #
            # #找出最小距离那个点
            # if distant < 90:
            #     toBeTrack[elemIndex][2] = tracked[elemTrackedindex][2]  # 是同一个东西 融合跟踪的ID和检测到的ID
            #     label[toBeTrack[elemIndex][3]] = tracked[elemTrackedindex][2]
            #
            #     cv2.circle(drawimg, (tracked[elemTrackedindex][0], tracked[elemTrackedindex][1]), 3, (0, 255, 255))
            #     cv2.circle(drawimg, (toBeTrack[elemIndex][0], toBeTrack[elemIndex][1]), 3, (255, 0, 255))
            #
            #     print("<90!!!!!!fusion,ID!!!===== ",  tracked[elemTrackedindex][2])
            #     剩下的点
            #     if distant > 150:
            #         toBeTrack[elemIndex][2] = maxNo+ak
            #         label[toBeTrack[elemIndex][3]] = maxNo+ak
            #         print("> 150!!!!!ADD CHANGE,ID ===== ", maxNo+ak)
            #         ak += 1
        return toBeTrack, label


    def detectVideo(self, yolo, output_path=""):
        """
        进行实时视频检测功能
        :param yolo: yolo实例对象
        :param output_path: 识别效果的视频保存位置，如不指定，默认为空
        :return: None，通过break跳出循环
        """
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        if self.cam._data_buf == -1 or self.cam._data_buf is None:
            raise IOError("Couldn't open webcam or video")
        # 视频编码格式
        video_FourCC = 6
        video_fps = 30
        video_size = (int(stFrameInfo.nWidth),
                      int(stFrameInfo.nHeight))
        isOutput = True if output_path != "" else False
        if isOutput:
            print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        while True:
            # print(cam._data_buf)
            frame = np.asarray(self.cam._data_buf)
            frame = frame.reshape((960, 1280, 3))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = PImage.fromarray(frame)  # PImage: from PIL import Vision as PImage
            # image.show()
            image = yolo.detectImage(image)

            result = np.asarray(image)
            curr_time = timer()
            exec_time = curr_time - prev_time  # 计算图像识别的执行时间
            prev_time = curr_time  # 重新设置时间节点
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1  # 计算每1s内FPS累加数
            if accum_time > 1:  # 累计时间超过1s，输出1s内处理的图片数（帧数）
                accum_time = accum_time - 1  # 累计时间超过1s后，重新开始统计
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0  # 时间超过1s后，清空fps数据，重新开始统计下一秒的帧率
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)  # 将数据写入到图像
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", result)
            if isOutput:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret = self.cam.MV_CC_GetOneFrameTimeout(byref(self.cam._data_buf), self.cam._nPayloadSize, stFrameInfo,
                                                    1000)
            if ret == 0:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                    stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
            else:
                print("no data[0x%x]" % ret)
            if g_bExit is True:
                break
        self.cam.destroy(self.cam, self.cam._data_buf)
        yolo.closeSession()


    def detectSerialImage(self, cam, transDict, transList, targetDict, transFrame, flag):
        """
        获取并处理连续的帧数
        :param cam: 相机对象
        :return: {"nFrame":nframe,"image":image, "timecost":timecost, "box":[(label1,xmin1,ymin1,xmax1, ymax1),(label2, xmin2, ymin2, xmax2, ymax2)]}
                返回检测到的物体类别、位置信息（xmin, ymin, xmax, ymax）, 识别耗时，原始帧数据返回（便于后续操作，eg：Draw the box real time）
        """
        global state
        prev_time = timer()
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        # if sysArc[0] == "Linux" and sysArc[-1] == "aarch64":
        #     print("press_any_key_exit!")
        #     cam.press_any_key_exit()

        # trackObj = ImageTrack()

        # avi = Video("E:\\1\\1.avi")
        # avi = cam
        # if avi is not None:
        #     preframe = avi.getImageFromVideo()
        # else:
        preframe, nFrame, t = cam.getImage()
        preframeT = t


        #保存一张测试图片
        # cv2.imwrite("e:\\1\\research.jpg", preframe)

        preframeb, bgMaskb, resarray = self.imgproc.delBg(preframe) if self.imgproc else (preframe, None)
        k = 1
        startt = timer()
        left = 0
        top = 0
        right = 0
        bottom = 0
        # flag = 0
        # preflag =0
        inputCorner = np.array([])
        p0 = np.array([])
        label = np.array([])
        # avi = Video("E:\\1\\1.avi")
        # frame = avi.getImageFromVideo()
        i = 1
        dataDict = {}
        startTime = 0
        # fig = plt.figure()
        plt.ion()  # 开启一个画图的窗口

        # ax = fig.add_subplot(1, 1, 1)

        x_ = []
        y_ = []
        idlist = []
        idXYDict = {}

        statelist = []
        frameTlist =[]
        bgtime = 0
        reckon =0
#数据融合通过它 都融合在这里面
        LKtrackedList= []
        while True:
            if flag[0] == 2:
                print("重新进行背景学习")
                if useCamra:
                    self.imgproc.reStudyBgModelFromCam(cam)
                else:
                    self.imgproc.studyBackground()
                    self.imgproc.createModelsfromStats()
                flag[0] = 0


            print("track State--------------->", state)

            _frame, nFrame, t = cam.getImage()
            frameT = t
            statelist.append(1000*bgtime)
            frameTlist.append(frameT)
            plt.clf()  # 清除之前画的图
            # plt.plot(frameTlist, statelist, 'rs--', label=str(0))  # 画出当前 ax 列表和 ay 列表中的值的图形
            # plt.pause(0.1)
            plt.ioff()
            deltaT = frameT - preframeT  #时间间隔
            camfps = "Cam" + cam.getCamFps(nFrame)
            # frame = avi.getImageFromVideo()
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "NetFPS:" + str(curr_fps)
                curr_fps = 0

            BgLearnTimeOBJ.calSet()
            frame, bgMask, resarray = self.imgproc.delBg(_frame) if self.imgproc else (_frame, None)
            bgtime = BgLearnTimeOBJ.calEnd()
            BgLearnTimeOBJ.printCostTime()
            result = frame.copy()
            if crop is False:
                img = PImage.fromarray(frame)  # PImage: from PIL import Vision as PImage
                YoloTimeOBJ.calSet()
                dataDict = self.yolo.detectImage(img)
                YoloTimeOBJ.calEnd()
                YoloTimeOBJ.printCostTime()

                result = np.asarray(dataDict["image"])

                dataDict["image"] = img  # img：Image对象
                dataDict["nFrame"] = nFrame
                dataDict["frameTime"] = t  # 相机当前获取打当前帧nFrame的时间t
                # print("bigPicTimeCost", dataDict["timeCost"])

            dataDict["bgTimeCost"] = self.imgproc.bgTimeCost if self.imgproc else 0


            # arr = np.asarray(dataDict["image"])
            imglist = self.imgproc.getBoxOnlyPic(dataDict, preframe)
            imglistk = self.imgproc.getBoxOnlyPic(dataDict, _frame)

            drawimg = frame.copy()
            featureimg = cv2.cvtColor(preframeb, cv2.COLOR_BGR2GRAY)
            secondimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detectbox = self.imgproc.filterBgBox(resarray, drawimg)
            objCNNList = []
            if crop is True:
                # print("detectbox.......", detectbox)
                # objCNNList= self.imgproc.cropBgBoxToYolo(detectbox, drawimg,  self.yolo, nFrame, t)
                # kk = 0
                # if len(objCNNList) > 0:
                #     for i in range(len(objCNNList)):
                #         kk += 1
                #         arrM = np.asarray(objCNNList[i]["image"])
                #         cv2.imshow(str(kk), arrM)
                pass


            # detect

            # dataDict box[predicted_class, score, left, top, right, bottom, \
            #                 angle, diameter, centerX, centerY, trackID, deltaX, deltaY, speedX, speedY]
            if flag[0] == 0:
                #clear the track idlist
                idlist = []
                idXYDict = {}
                if crop is False:
                    DetectTimeOBJ.calSet()
                    p0, label, toBeTrackedList, centerList, dataDict = self.imgproc.detectObj(featureimg, drawimg, dataDict)
                    DetectTimeOBJ.calEnd()
                    DetectTimeOBJ.printCostTime()

                    dataDict["frameTime"] = frameT
                    # if dataDict is not None and "box" in dataDict:
                    #     print("dataDict:", dataDict)
                    #     for i in range(len(dataDict["box"])):
                    #         print("%%%%%%current i", i)
                    #         if dataDict["box"][i][10] is not None:
                    #             print("%%%%%%%%currentID", dataDict["box"][i][10])
                    #             print("now in detect!!!!!!!!!!!!!!!!!!!!!!")


                    if toBeTrackedList is not None:
                        for seqN in range(len(toBeTrackedList)):
                            idlist.append(toBeTrackedList[seqN][2])
                        # print("detect 88888888888 idlist", idlist)
                        for idElem in idlist:
                            idXYDict[idElem] = [[], []]  # x, y坐标

                else:
                    centerList = self.imgproc.detectObjNotRelyLK(featureimg, drawimg, objCNNList)

               #dataDict就是检测到的，centerX, centerY, trackID三个变量已经附加上
                # centerList不再需要，仍旧保留，等待以下代码更改为dataDict
                # boxList.append([predicted_class, score, left, top, right, bottom, \
                #                 angle, diameter, centerX, centerY, trackID, deltaX, deltaY, speedX, speedY])

                # print("centerList", centerList)

                # tempList = centerList

                if centerList is not None and len(centerList) > 0:
                    # transList = [[] for j in range(len(centerList))]
                    if len(transList) == 0:
                        for seqN in range(len(centerList)):
                            transList.append(centerList[seqN])

                    elif len(transList) <= len(centerList):
                        deltaCnt = len(transList)
                        for seqN in range(len(centerList)):
                            if deltaCnt > 0:
                                transList.pop(0)
                                deltaCnt -= 1
                            # print("seqN-----------", seqN)
                            transList.append(centerList[seqN])
                        # print("666666", transList)

                    else:
                        # transList = []  千万不能这样操作！
                        deltaCnt = len(transList)
                        for seqN in range(len(transList)):
                            if deltaCnt > 0:
                                transList.pop(0)
                                deltaCnt -= 1
                            if 0 <= seqN - (len(transList) - len(centerList)) < len(centerList):
                                # print("seqN-----------", seqN - (len(transList) - len(centerList)))
                                transList.append(centerList[seqN - (len(transList) - len(centerList))])
                        # print("55555555", transList)
                                # transList.append([])

                    #     print(transList, centerList, str(len(transList)), str(len(centerList)))
                    #     print(len(centerList[seqN]), len(transList[seqN]))
                    #     for jj in range(len(centerList[seqN])):
                    #         transList[seqN].append(centerList[seqN][jj])


                    # for seqN in range(len(centerList)):
                    #     cv2.circle(drawimg, (int(centerList[seqN][0]), int(centerList[seqN][1])), 24, (0, 0, 255), 7)
                    #     cv2.putText(drawimg, text=str(int(centerList[seqN][2])),
                    #                 org=(int(centerList[seqN][0]) - 20, int(centerList[seqN][1])),
                    #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    #                 fontScale=2, color=(255, 255, 255), thickness=2)
                    print("########################")
                    print("centerList", centerList)
                    print("transList", transList)
                    print("########################")
                    # if centerList[seqN][3] == 0 or centerList[seqN][4] == 0:
                        #     centerList = []
                        #     transList = centerList
                #切换为图像跟踪模式
                if toBeTrackedList is not None:
                    print("toBeTrackedList", np.array(toBeTrackedList)[:, 0:3])

                if p0 is not None and label is not None and toBeTrackedList is not None:
                    if toBeTrackedList is not None and LKtrackedList is not None and len(LKtrackedList) > 0:
                        toBeTrackedList, label = self.dataFusion(toBeTrackedList, LKtrackedList, label)
                        # cv2.waitKey(2000)
                    flag[0] = 1



            # track
            else:
                # LKtrackedList中保存的是 被跟踪到的点的 位置坐标，位移，和位移除以时间后的速度，格式如下
                # LKtrackedList[seqN][0]    centerX
                # LKtrackedList[seqN][1]    centerY
                # LKtrackedList[seqN][2]    trackID
                # LKtrackedList[seqN][3]    和label对应的p0的 index
                # LKtrackedList[seqN][4]    deltaX
                # LKtrackedList[seqN][5]    deltaY
                # LKtrackedList[seqN][6]    speedX
                # LKtrackedList[seqN][7]    speedY


               # frameT 传入作为跟踪起始时间 如果和 dataDict[frameTime] 相等，
                # 则代表跟踪的是dataDict[frameTime] 那个时间的dataDict的数据
                #代表第一次进
                # if preflag != flag:
                #     startTime = frameT
                # trackStartTime = 0
                # trackEndTime = 0
                # if statisticTrackTime is True:
                #     trackStartTime = timer()
                LkTrackTimeOBJ.calSet()
                p0, label, LKtrackedList, state = self.imgproc.trackObj(featureimg, secondimg, drawimg, label, p0, deltaT)
                # if statisticTrackTime is True:
                #     trackEndTime = timer()
                # trackCostTime = trackEndTime - trackStartTime
                LkTrackTimeOBJ.calEnd()
                LkTrackTimeOBJ.printCostTime()
                # print("TrackCostTime!!!!!!!!!!!!!!!!!!!!!! = %f ms" % (trackCostTime*1000))

                # print("!!!!!!!!!!!tracked p0 = ", p0)
                # print("!!!!!!!!!!!!tracked label = ", label)
                if LKtrackedList is not None and len(LKtrackedList) > 0:
                    print("LKtrackedList", np.array(LKtrackedList)[:, 0:3])

                if LKtrackedList is not None and len(LKtrackedList) > 0:
                    # LKtrackedList[seqN][0]    centerX
                    # LKtrackedList[seqN][1]    centerY
                    # ax.plot([1, 2, 3, 4], [2, 3, 4, 5])
                    print("LKtrackedList~~~~~~~~~:", LKtrackedList)
                    print("in the Track!!!!!!!!!!!!!!!!", LKtrackedList)


                    for seqN in range(len(LKtrackedList)):

                        # print("LKtrackedList########################", LKtrackedList)
                        print("111111111111111111111111", transList, LKtrackedList)

                        # uuIDText = targetDict["target"][seqN][0]
                        # 位置坐标
                        cv2.circle(drawimg, (int(LKtrackedList[seqN][0]), int(LKtrackedList[seqN][1])), 24, (0, 0, 255), 5)
                        # cv2.circle(drawimg, (int(targetDict["target"][seqN][2][0]),
                        #                      int(targetDict["target"][seqN][2][1])), 6, (0, 0, 200), 2)
                        # cv2.putText(drawimg, uuIDText, (int(targetDict["target"][seqN][2][0]) + 50,
                        #                                 int(targetDict["target"][seqN][2][1]) + 50),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        # ID 和 偏移量
                        cv2.putText(drawimg, text=str(int(LKtrackedList[seqN][4])),
                                    org=(LKtrackedList[seqN][0] - 50, LKtrackedList[seqN][1] - 30),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=2, color=(255, 255, 0), thickness=2)
                        cv2.putText(drawimg, text=str(int(LKtrackedList[seqN][5])),
                                    org=(LKtrackedList[seqN][0] + 50, LKtrackedList[seqN][1] - 30),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=2, color=(255, 255, 0), thickness=2)
                        cv2.putText(drawimg, text="ID:",
                                    org=(LKtrackedList[seqN][0], LKtrackedList[seqN][1] - 50),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=2, color=(0, 255, 255), thickness=2)
                        cv2.putText(drawimg, text=str(LKtrackedList[seqN][2]),
                                    org=(LKtrackedList[seqN][0]+50, LKtrackedList[seqN][1]-50),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=2, color=(0, 255, 255), thickness=2)
                        # speed
                        cv2.putText(drawimg, text=str(int(LKtrackedList[seqN][6])),
                                    org=(LKtrackedList[seqN][0] - 50, LKtrackedList[seqN][1]+30),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=2, color=(255, 255, 0), thickness=2)
                        cv2.putText(drawimg, text=str(int(LKtrackedList[seqN][7])),
                                    org=(LKtrackedList[seqN][0] + 50, LKtrackedList[seqN][1] + 30),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=2, color=(255, 255, 0), thickness=2)

                        # if centerList[seqN][3] == 0 or centerList[seqN][4] == 0:
                        #     centerList = []
                        #     transList = centerList


                    if len(transList) == 0:
                        for seqN in range(len(LKtrackedList)):
                            transList.append(LKtrackedList[seqN])

                    elif len(transList) <= len(LKtrackedList):
                        deltaCnt = len(transList)
                        for seqN in range(len(LKtrackedList)):
                            if deltaCnt > 0:
                                transList.pop(0)
                                deltaCnt -= 1
                            print("seqN-----------", seqN)
                            transList.append(LKtrackedList[seqN])
                        print("666666", transList)

                    else:
                        # transList = []  千万不能这样操作！
                        deltaCnt = len(transList)
                        for seqN in range(len(transList)):
                            if deltaCnt > 0:
                                transList.pop(0)
                                deltaCnt -= 1
                            if 0 <= seqN - (len(transList) - len(LKtrackedList)) < len(LKtrackedList):
                                print("seqN-----------", seqN - (len(transList) - len(LKtrackedList)))
                                transList.append(LKtrackedList[seqN - (len(transList) - len(LKtrackedList))])
                        print("55555555", transList)



                    # try to transfer the frame
                    # transFrame = np.zeros((10, 15, 3), np.uint8)

                    #遍历所有的跟踪ID 构造ID字典 存放跟踪到的坐标序列
                    # color = ["r", "g", "b"]
                    # if len(idlist) > 0:
                    #     print("in track88888888888 idlist", idlist)
                    #     plt.clf()  # 清除之前画的图
                    #     for idElem in idlist:
                    #         # idXYDict[idElem] = [[], []]  # x, y坐标
                    #         for seqN in range(len(LKtrackedList)):
                    #             if LKtrackedList[seqN][2] == idElem:
                    #                 # x_.append(LKtrackedList[seqN][0])
                    #                 # y_.append(LKtrackedList[seqN][1])
                    #                 idXYDict[idElem][0].append(LKtrackedList[seqN][0])
                    #                 idXYDict[idElem][1].append(LKtrackedList[seqN][1])
                    #                 # plt.plot(idXYDict[idElem][0], idXYDict[idElem][1], color[int(idElem)%3]+'s--', label=str(int(idElem)))  # 画出当前 ax 列表和 ay 列表中的值的图形
                    #                 # plt.pause(0.1)  # 暂停一秒
                    #     print("idXYDict!!!!!!!!", idXYDict)
                    #     for key in idXYDict.keys():
                    #         print("DICT KEY,PLOT:", key)


                        # plt.ioff()  # 关闭画图的窗口 没有的话会内存溢出

                    print("@" * 100)
                    # print(len(frame))
                    for l in range(6):
                        for ll in range(7):
                            for lll in range(3):
                                pass
                                # transFrame[l][ll][lll] = frame[LKtrackedList[0][0] + l][LKtrackedList[0][1] + ll][lll]

                                # transFrame[l][ll] = [1, 2, 3]
                            # print(transFrame[l][ll])
                    # print(centerList)
                    # print(transFrame)
                    # print(frame[centerList[0][0]][centerList[0][1]])
                    # cv2.imshow("frame", frame)
                    print("@" * 100)
                    a = np.array(x_)
                    b = np.array(y_)
                    a = a / a.mean(axis=0)
                    b = b / b.mean(axis=0)

                    # plt.plot([LKtrackedList[0][0]], [LKtrackedList[0][1]])
                    # print("x~~~~~~~~~~~", a)
                    # print("y~~~~~~~~~~~", b)
                    # plt.pause(0.1)
                    # plt.clf()
                    # plt.show()

            # clear

            # if centerList:
            #     0
            # if dataDict is not None and "box" in dataDict:
            #     print("dataDict!!!:", dataDict)
            # 如果CNN没识别到瓶子，则跳回detect
            if "box" not in dataDict:
                # p0 = np.array([])
                # label = np.array([])
                # flag = 0
                cv2.circle(drawimg, (100, 100), 15, (0, 0, 255), -1)  # red  track

            else:   #如果至少有一个瓶子置信度高 就不跳回detect 否则跳回ditect
                nonBottleFlag = True
                for x in range(len(dataDict["box"])):
                    if dataDict["box"][x][1] > 0.9:
                        nonBottleFlag = False
                        break

                if nonBottleFlag is True:
                    # p0 = np.array([])
                    # label = np.array([])
                    # flag = 0
                    cv2.circle(drawimg, (100, 100), 15, (0, 0, 255), -1)  # red  track
            cv2.imshow("res", drawimg)
            cv2.waitKey(10)
            preframeb = frame.copy()
            preframeT = frameT
            # preflag = flag

            if bgMask is not None:
                dataDict = self.imgproc.getBottlePose(_frame, bgMask, dataDict)
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.putText(result, text=camfps, org=(150, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(0, 255, 255), thickness=2)
            cv2.imshow("result", result)
            # cv2.waitKey(1000)
            cv2.waitKey(10)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cam.destroy()
                break
                #重新开始检测  如果时间大于1s
            if frameT - reckon > 1:
                reckon = frameT
                # flag[0] = 0   先屏蔽掉

            if cv2.waitKey(1) & 0xFF == ord('r'):
                flag[0] = 0
                print("change!!!! flag=0!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                cv2.circle(drawimg, (100, 100), 15, (0, 255, 255), -1)  # red  track
            # return dataDict
            global bottleDict
            bottleDict = dataDict
            # bottleDict1 = dataDict
            # 此处不能直接使用transDict = dataDict,不然其他进程读取不到数据
            transDict.update(dataDict)
            # print("*" * 100)
            # print(transDict)
            # print("*" * 100)
            # print(transDict)
            # print(bottleDict1)
            # print(bottleDict)
            bottleDict1 = dataDict
            # transDict = {}
            # if "box" in dataDict:
            #     transDict["box"] = bottleDict["box"]
            transFrame = copy.deepcopy(frame)
            # transDict = {"---------------------------------------------------" + str(i)}
            # print(transDict)
            # print(bottleDict["box"])
            # print(dict)
            i += 1
        cam.destroy()

    def detectSingleImage(self, frame, nFrame):
        """
        用于接受bgLearn返回过来的图片
        :param frame: opencv格式的图片，例如：frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        :param nFrame: 图片的帧号，用来确定图像的唯一性
        :return: {"nFrame":nframe,"image":image, "timecost":timecost, "box":[(label1,xmin1,ymin1,xmax1, ymax1),(label2, xmin2, ymin2, xmax2, ymax2)]}
                返回检测到的物体类别、位置信息（xmin, ymin, xmax, ymax）, 识别耗时，原始帧数据返回（便于后续操作，eg：Draw the box real time）
        """
        # cv2.namedWindow("kk", cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("kk", frame)
        # cv2.waitKey(3000)
        # 设定计时器, 统计识别图像耗时
        prev_time = timer()
        # 将opencv格式的图像数据转换成PIL类型的image对象，便于进行标框和识别效果可视化
        img = PImage.fromarray(frame)  # PImage: from PIL import Vision as PImage
        # img.show()
        # feed data into model
        dataDict = self.yolo.detectImage(img)
        curr_time = timer()
        exec_time = curr_time - prev_time
        # dataDict["timecost"] = exec_time
        dataDict["nFrame"] = nFrame
        arr = np.asarray(dataDict["image"])
        cv2.imshow("result", arr)
        # cv2.waitKey(1000)
        cv2.waitKey(10)
        return dataDict


"""
if __name__ == '__main__':
    cam = Camera()
    _frame, nf = cam.getImage()
    print("准备载入yolo网络！")
    yolo = YOLO()q
    _image = Vision(cam, yolo)
    dataDict = _image.detectSerialImage(_frame, nf)
    print(dataDict)
    # image.detectVideo(yolo)
"""


def imageInit():
    """
    初始化相机对象cam, Vision对象
    :return: (cam：相机对象, _image:Vision对象)
    """
    cam = None
    avi =None
    bgAvi =None

    if useCamra:
        cam = Camera()
    else:
        avi = Video(videoDir)
        bgAvi = Video(bgDir)
    if useCamra:
        imgCapObj = imageCapture(cam, None, cam)
    else:
        imgCapObj = imageCapture(None, avi, bgAvi)

    # _frame, nf = cam.getImage()
    print("准备载入yolo网络！")
    yolo = YOLO()
    print("准备背景学习！")
    bgobj = ImgProc(50, imgCapObj)

    # bgobj.studyBackground()
    # bgobj.createModelsfromStats()
    # 重新学习背景 每次需要重新学习背景的时候，就调用这个方法
    if useCamra:
        bgobj.reStudyBgModelFromCam(cam)
    else:
        bgobj.studyBackground()
        bgobj.createModelsfromStats()
    _image = Vision(imgCapObj, yolo, bgobj)
    print("开始！")
    global gState
    gState = 2
    return imgCapObj, _image


"""
def imageInit():
    cam = Camera()
    # _frame, nf = cam.getImage()
    print("准备载入yolo网络！")
    yolo = YOLO()
    print("准备背景学习！")
    bgobj = ImgProc(50)
    # bgobj.studyBackgroundFromCam(cam)
    bgobj.studyBackgroundFromVideo("E:\\1\\背景.avi")
    bgobj.createModelsfromStats()
    _image = Vision(cam, yolo, bgobj)
    print("开始！")
    global gState
    gState = 2
    return cam, _image
"""


def imageRun(cam, _image, transDict, transList, targetDict, transFrame, Flag):
    """
    根据输入的图像数据，进行识别
    :param cam: 相机对象
    :param _image: Vision对象
    :return: None | 系统有异常，退出系统
    """
    # while 1:
    #     try:
    #         _frame, nf = cam.getImage()
    #         frameDelBg = _image.bgLearn.delBg(_frame)
    # print(transDict)

    _image.detectSerialImage(cam, transDict, transList, targetDict, transFrame, Flag)

    # dataDict["bgTimeCost"] = _image.bgLearn.bgTimeCost
    # cv2.waitKey(10)
    #         print(dataDict)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     except Exception as e:
    #         global gState
    #         gState = 3
    #         print(e)
    #         break
    # cam.destroy()
    print("系统退出中···")
    sys.exit()


# 将imageInit()和imageRun()封装成一个函数，才能在一个进程中使用
def vision_run(transDict, transList, targetDict, transFrame, Flag):
    cam, _image = imageInit()
    # # while 1:
    # transDict["aaa"] = 666666
    imageRun(cam, _image, transDict, transList, targetDict, transFrame, Flag)
"""
if __name__ == '__main__':
    cam = Camera()
    # _frame, nf = cam.getImage()
    print("准备载入yolo网络！")
    yolo = YOLO()
    print("准备背景学习！")
    bgobj = Bglearn()
    bgobj.studyBackgroundFromCam(cam)
    bgobj.createModelsfromStats(6.0)
    _image = Vision(cam, yolo, bgobj)
    print("开始！")
    while 1:
        try:
            _frame, nf = cam.getImage()
            frameDelBg = _image.bgLearn.delBg(_frame)
            dataDict = _image.detectSerialImage(frameDelBg, nf)
            dataDict["bgTimeCost"] = _image.bgLearn.bgTimeCost
            #cv2.waitKey(10)
            print(dataDict)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(e)
            break
    cam.destroy()
"""


def video_run(transDict, transList, targetDict, transFrame, Flag):
    # cam, _image = imageInit()
    # cam = Camera()
    videoDir = "E:\\1\\1\\鲁\\Video_20200204122733339.mp4"
    bgDir = "E:\\1\\1\\鲁\\背景1.avi"

    # videoDir = "d:\\1\\Video_20200204122733339.mp4"
    # bgDir = "d:\\1\\背景1.avi"

    # videoDir = "E:\\1\\两个瓶子.avi"
    # bgDir = "E:\\1\\一个瓶子背景.avi"

    avi = Video(videoDir)
    bgAvi = Video(bgDir)
    imgCapObj = imageCapture(None, avi, bgAvi)
    # imgCapObj = imageCapture(cam, None, cam)

    # _frame, nf = cam.getImage()
    print("准备载入yolo网络！")
    yolo = YOLO()
    print("准备背景学习！")
    bgobj = ImgProc(50, imgCapObj)
    # bgobj.studyBackgroundFromCam(cam)
    # bgobj.studyBackgroundFromCam(bgAvi)
    bgobj.studyBackground()
    bgobj.createModelsfromStats()
    _image = Vision(imgCapObj, yolo, bgobj)
    print("开始！")
    global gState
    gState = 2
    imageRun(imgCapObj, _image, transDict, transList, targetDict, transFrame, Flag)


# def read(transDict):
#     while True:
#         print("=" * 100)
#         print(transDict)
#         print("=" * 100)
#         time.sleep(1)


def read(transDict, transList):
    while True:
        print("*******")
        print(transDict)
        print(transList)
        print("*******")
        time.sleep(0.5)
    # print('Process to read: %s' % os.getpid(), time.time())
    # while True:
    #     value = bottlDict1.get(True)
    #     print(value)
    #
    #     # print('Get %s from dict.  ---- currentTime:' % value, time.time())


if __name__ == '__main__':
    # # cam = Vision()
    # with multiprocessing.Manager() as MG:  # 重命名
    #     transDict = MG.dict()
    #     transList = MG.list()
    #     targetDict = MG.dict()
    #     # transFrame = MG.Array("i", range(126))
    #     # transFrame = MG.Array("i", np.zeros((6, 7, 3), np.uint8))
    #     transFrame = multiprocessing.RawArray('d', np.zeros((6, 7, 3), np.double).ravel())
    #     # cam, _image = imageInit()
    #     p2 = multiprocessing.Process(target=read, args=(transDict, transList,))
    #     p2.daemon = True
    #     p2.start()
    #     # p2.join()
    #     p1 = multiprocessing.Process(target=vision_run, args=(transDict, transList, targetDict, transFrame))
    #     p1.daemon = True
    #     p1.start()
    #     p1.join()
    #     # _image.detectSerialImage(cam, transDict, )
    #
    #     # pw = multiprocessing.Process(target=imageRun, args=(cam, _image, transDict))
    #     # pw.run()
    #     # pw.join()
    #     # imageRun(cam, _image, transDict)
    #
    #     # by yuantao1880@126.com
    #     # with multiprocessing.Manager() as MG:  # 重命名
    #     #     transDict = MG.dict()
    #     #     p2 = multiprocessing.Process(target=read, args=(transDict,))
    #     #     # 设置进程守护，主进程停止后，子进程也停止
    #     #     p2.daemon = True
    #     #     p2.start()
    #     #     # 开启一个子进程，进行识别跟踪
    #     #     pw = multiprocessing.Process(target=vision_run, args=(transDict,))
    #     #     pw.daemon = True
    #     #     pw.start()
    #     #     p2.join()

    track = Track()
    trackFlag = 0

    with multiprocessing.Manager() as MG:  # 重命名

        transDict = MG.dict()
        transList = MG.list()
        targetDict = MG.dict()
        Flag = MG.list()
        if len(Flag) == 0:
            Flag.append(0)
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

        # first line code is without filter; second line one is with filter
        # p2 = multiprocessing.Process(target=track.trackProcess, args=(transDict, transList, targetDict))
        # # p2 = multiprocessing.Process(target=track.trackWithFilter, args=(transDict, transList, targetDict))
        # p2.daemon = True
        # p2.start()

        p1 = multiprocessing.Process(target=vision_run, args=(transDict, transList, targetDict, transFrame,Flag))
        p1.daemon = True  #主进程运行完不会检查p1子进程的状态（是否执行完），直接结束进程；
        p1.start()
        p1.join()  #阻塞当前进程p2，直到p1执行完，再执行p2