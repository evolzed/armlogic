import os
import sys

from src.BS02.track.track import *

sys.path.append(os.path.abspath("../../"))
import numpy as np
from ctypes import *
from timeit import default_timer as timer
import cv2
from src.BS02.imageProcess.imgProc import *
from src.BS02.video import Video
from src.BS02.interface import imageCapture
import time
import multiprocessing
# sys.path.insert(0, os.path.split(__file__)[0])
# from lib.GrabVideo import GrabVideo
import platform
import copy
from lib.Logger.Logger import Logger

# sys.stdout = Logger("d:\\12.txt")  # 保存到D盘

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
            frame = np.asarray(cam._data_buf)
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
            ret = self.cam.MV_CC_GetOneFrameTimeout(byref(cam._data_buf), cam._nPayloadSize, stFrameInfo, 1000)
            if ret == 0:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                    stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
            else:
                print("no data[0x%x]" % ret)
            if g_bExit is True:
                break
        self.cam.destroy(self.cam, cam._data_buf)
        yolo.closeSession()

    def detectSerialImage(self, cam, transDict, transList, targetDict, transFrame):
        """
        获取并处理连续的帧数
        :param cam: 相机对象
        :return: {"nFrame":nframe,"image":image, "timecost":timecost, "box":[(label1,xmin1,ymin1,xmax1, ymax1),(label2, xmin2, ymin2, xmax2, ymax2)]}
                返回检测到的物体类别、位置信息（xmin, ymin, xmax, ymax）, 识别耗时，原始帧数据返回（便于后续操作，eg：Draw the box real time）
        """


        prev_time = timer()
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        # if sysArc[0] == "Linux" and sysArc[-1] == "aarch64":
        #     print("press_any_key_exit!")
        #     cam.press_any_key_exit()

        # trackObj = ImageTrack()

        # avi = Video("E:\\1\\1.avi")
        # preframe = avi.getImageFromVideo()
        preframe, nFrame, t = cam.getImage()
        preframeb, bgMaskb, resarray = self.imgproc.delBg(preframe) if self.imgproc else (preframe, None)
        k = 1
        startt = timer()
        left = 0
        top = 0
        right = 0
        bottom = 0
        flag = 0
        inputCorner = np.array([])
        p0 = np.array([])
        label = np.array([])
        # avi = Video("E:\\1\\1.avi")
        # frame = avi.getImageFromVideo()
        i = 1
        while True:
            _frame, nFrame, t = cam.getImage()
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

            frame, bgMask, resarray = self.imgproc.delBg(_frame) if self.imgproc else (_frame, None)

            # transFrame = np.zeros((4, 5, 3), np.uint8)
            # print("@" * 100)
            # print(len(frame))
            # for l in range(4):
            #     for ll in range(5):
            #         for lll in range(3):
            #             transFrame[l][ll][lll] = frame[l][ll][lll]
            #             # print(frame)
            # print("@" * 100)

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
            dataDict = self.yolo.detectImage(img)
            dataDict["bgTimeCost"] = self.imgproc.bgTimeCost if self.imgproc else 0
            result = np.asarray(dataDict["image"])
            # dataDict["image"] = result  # result：cv2.array的图像数据
            dataDict["image"] = img  # img：Image对象
            # dataDict["timeCost"] = exec_time
            dataDict["nFrame"] = nFrame
            dataDict["frameTime"] = t  # 相机当前获取打当前帧nFrame的时间t
            # arr = np.asarray(dataDict["image"])
            imglist = self.imgproc.getBoxOnlyPic(dataDict, preframe)
            imglistk = self.imgproc.getBoxOnlyPic(dataDict, _frame)
            drawimg = frame.copy()
            featureimg = cv2.cvtColor(preframeb, cv2.COLOR_BGR2GRAY)
            secondimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detectbox = self.imgproc.filterBgBox(resarray, drawimg)
            # centerlist = centerList = None
            # detect
            if flag == 0:
                p0, label, centerlist = self.imgproc.detectObj(featureimg, drawimg, dataDict, 3)
                # p0, label, centerlist = self.imgproc.detectObjNotRelyCnn(featureimg, drawimg, detectbox, 3)
                # print("########################", centerlist)
                # print(transList)
                if centerlist is not None and len(centerlist) > 0:
                    # transList = [[] for j in range(len(centerlist))]
                    # print(transList, centerlist, str(len(transList)), str(len(centerlist)))
                    for seqN in range(len(centerlist)):
                        transList.append(centerlist[seqN])
                    #     print(transList, centerlist, str(len(transList)), str(len(centerlist)))
                    #     print(len(centerlist[seqN]), len(transList[seqN]))
                    #     for jj in range(len(centerlist[seqN])):
                    #         transList[seqN].append(centerlist[seqN][jj])
                        cv2.circle(drawimg, (centerlist[seqN][0], centerlist[seqN][1]), 24, (0, 0, 255), 7)
                    # print(centerlist, transList)
                        # if centerlist[seqN][3] == 0 or centerlist[seqN][4] == 0:
                        #     centerlist = []
                        #     transList = centerlist
                if p0 is not None and label is not None:
                    flag = 1

            # track
            else:
                p0, label, centerList = self.imgproc.trackObj(featureimg, secondimg, drawimg, label, p0)
                if centerList is not None and len(centerList) > 0:
                    for seqN in range(len(centerList)):
                        print("########################", centerList)
                        # transList = [[] for j in range(len(centerList))]
                        # for jj in range(len(centerList[seqN])):
                        #     transList[seqN].append(centerList[seqN][jj])
                        # transList.append(centerList[seqN])
                        print("111111111111111111111111", transList, centerList)
                        transList[seqN] = centerList[seqN]
                        # uuIDText = targetDict["target"][seqN][0]
                        cv2.circle(drawimg, (centerList[seqN][0], centerList[seqN][1]), 24, (255, 0, 0), 7)
                        # cv2.circle(drawimg, (int(targetDict["target"][seqN][2][0]),
                        #                      int(targetDict["target"][seqN][2][1])), 6, (0, 0, 200), 2)
                        # cv2.putText(drawimg, uuIDText, (int(targetDict["target"][seqN][2][0]) + 50,
                        #                                 int(targetDict["target"][seqN][2][1]) + 50),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(drawimg, text=str(int(centerList[seqN][3])),
                                    org=(centerList[seqN][0] - 20, centerList[seqN][1]),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=2, color=(255, 255, 255), thickness=2)
                        cv2.putText(drawimg, text=str(int(centerList[seqN][4])),
                                    org=(centerList[seqN][0] - 20, centerList[seqN][1] + 50),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=2, color=(255, 255, 255), thickness=2)
                        cv2.putText(drawimg, text=str(centerList[seqN][2]),
                                    org=(centerList[seqN][0], centerList[seqN][1]),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=3, color=(0, 255, 255), thickness=2)

                        # if centerList[seqN][3] == 0 or centerList[seqN][4] == 0:
                        #     centerList = []
                        #     transList = centerList

                    # try to transfer the frame
                    # transFrame = np.zeros((10, 15, 3), np.uint8)

                    print("@" * 100)
                    print(len(frame))
                    for l in range(6):
                        for ll in range(7):
                            for lll in range(3):
                                transFrame[l][ll][lll] = frame[centerList[0][0] + l][centerList[0][1] + ll][lll]
                                # transFrame[l][ll] = [1, 2, 3]
                            print(transFrame[l][ll])
                    # print(centerList)
                    # print(transFrame)
                    # print(frame[centerList[0][0]][centerList[0][1]])
                    # cv2.imshow("frame", frame)
                    print("@" * 100)

            # clear

            # if centerList:
            #     0

            if "box" not in dataDict:
                p0 = np.array([])
                label = np.array([])
                flag = 0
                cv2.circle(drawimg, (100, 100), 15, (0, 0, 255), -1)  # red  track

            else:
                nonBottleFlag = True
                for x in range(len(dataDict["box"])):
                    if dataDict["box"][x][1] > 0.9:
                        nonBottleFlag = False
                        break

                if nonBottleFlag is True:
                    p0 = np.array([])
                    label = np.array([])
                    flag = 0
                    cv2.circle(drawimg, (100, 100), 15, (0, 0, 255), -1)  # red  track
            cv2.imshow("res", drawimg)
            cv2.waitKey(10)
            preframeb = frame.copy()

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
                break
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
    yolo = YOLO()
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
    cam = Camera()
    # videoDir = "d:\\1\\Video_20200204122733339.mp4"
    # bgDir = "d:\\1\\背景1.avi"
    # avi = Video(videoDir)
    # bgAvi = Video(bgDir)
    imgCapObj = imageCapture(cam, None, None)
    # imgCapObj = imageCapture(cam, None, cam)

    # _frame, nf = cam.getImage()
    print("准备载入yolo网络！")
    yolo = YOLO()
    print("准备背景学习！")
    bgobj = ImgProc(50, imgCapObj)
    # bgobj.studyBackgroundFromCam(cam)
    bgobj.studyBackgroundFromCam(cam)
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


def imageRun(cam, _image, transDict, transList, targetDict, transFrame):
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

    _image.detectSerialImage(cam, transDict, transList, targetDict, transFrame)

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
def vision_run(transDict, transList, targetDict, transFrame):
    cam, _image = imageInit()
    # # while 1:
    # transDict["aaa"] = 666666
    imageRun(cam, _image, transDict, transList, targetDict, transFrame)
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
        p2 = multiprocessing.Process(target=track.trackProcess, args=(transDict, transList, targetDict))
        # p2 = multiprocessing.Process(target=track.trackWithFilter, args=(transDict, transList, targetDict))
        p2.daemon = True
        p2.start()

        p1 = multiprocessing.Process(target=vision_run, args=(transDict, transList, targetDict, transFrame))
        p1.daemon = True
        p1.start()
        p1.join()