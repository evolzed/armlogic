import os
import sys
import numpy as np
from ctypes import *
from timeit import default_timer as timer
import cv2
from src.Vision.imageProcess.imgProc import ImgProc
sys.path.append(os.path.abspath("../../"))
# sys.path.insert(0, os.path.split(__file__)[0])
# from lib.GrabVideo import GrabVideo
import platform

from lib.Logger.Logger import Logger

sys.stdout = Logger("E:\\12.txt")  # 保存到D盘

sysArc = platform.uname()
if sysArc[0] == "Windows":
    from lib.HikMvImport_Win.utils.CameraParams_header import MV_FRAME_OUT_INFO_EX
elif sysArc[0] == "Linux" and sysArc[-1] == "aarch64":
    from lib.HikMvImport_TX2.utils.CameraParams_header import MV_FRAME_OUT_INFO_EX
else:
    print("不支持的系统架构，仅支持win10_64 和 Ubuntu16.04 ARM aarch64！！")
    sys.exit()
from src.Vision.camera import Camera, g_bExit
from src.Vision.yolo.Yolo import *
# from src.Vision.imageProcess.bgLearn import Bglearn
# from src.Vision.imageProcess.imageTrack import ImageTrack
gState = 1
bottleDict = None


class Vision(object):
    """create main Vision class for processing images"""

    def __init__(self, cam, yolo, imgproc_=None):
        """相机自检"""
        self.cam = cam
        self.yolo = yolo
        self.imgproc=imgproc_
        # self.deviceNum = cam.getDeviceNum()
        # cam._data_buf, cam._nPayloadsize = self.cam.connectCam()
        if -1 == cam._data_buf:
            print("相机初始化失败！退出程序！")
            sys.exit()
        print("相机初始化完成！")

    def detectVideo(self, yolo, output_path=""):
        """
        进行实时视频检测功能

        :param yolo: yolo实例对象
        :param output_path: 识别效果的视频保存位置，如不指定，默认为空
        :return: None，通过break跳出循环
        """
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        if cam._data_buf == -1 or cam._data_buf is None:
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

    def detectSerialImage(self, cam):
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

        #trackObj = ImageTrack()
        preframe, nFrame, t = cam.getImage()
        preframeb, bgMaskb, resarrayb = self.imgproc.delBg(preframe) if self.imgproc else (preframe, None)
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
        # good_old = np.array([])
        # p0 = np.array([])
        left = 0
        top = 0
        right = 0
        bottom = 0
        target=[]
        idd =0
        p0 = np.array([])
        p0_con =np.array([])
        pk = np.array([])
        input = np.array([])
        label = np.array([])
        while True:
            # try:
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

            frame, bgMask, resarray = self.imgproc.delBg(_frame) if self.imgproc else (_frame, None)
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
            # good_new, good_old, offset, img = self.img
            # proc.lkLightflow_track(imglist[0], imglistk[0], None)
            drawimg = frame.copy()
            featureimg = cv2.cvtColor(preframeb, cv2.COLOR_BGR2GRAY)
            secondimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #detect
            if flag == 0:
                p0 = cv2.goodFeaturesToTrack(featureimg, mask=None, **feature_params)

                if p0 is not None and "box" in dataDict:

                    for k in range(p0.shape[0]):
                        a = int(p0[k, 0, 0])
                        b = int(p0[k, 0, 1])
                        cv2.circle(drawimg, (a, b), 3, (0, 255, 255), -1)

                    label = np.ones(shape=(p0.shape[0], 1, 1), dtype=p0.dtype) * (-1)
                    print("len(dataDict[box])", len(dataDict["box"]))
                    boxLenth = len(dataDict["box"])
                    if boxLenth > 1:
                        for i in range(len(dataDict["box"])):
                            if "box" in dataDict and dataDict["box"][i][1] > 0.9 and dataDict["box"][i][3] > 180:
                                print("in!!!!!!!!!!!!!!!!!!!!!!!!!in!!!!!!!!!!!!!!!")
                                left = dataDict["box"][i][2]
                                top = dataDict["box"][i][3]
                                right = dataDict["box"][i][4]
                                bottom = dataDict["box"][i][5]
                                cv2.rectangle(drawimg, (left, top), (right, bottom), (255, 255, 0))
                                #store every point label
                                print("iiiiiiiiiiiiiiiiiiiiiiiiii------------:", i)
                                for k in range(p0.shape[0]):
                                    print("p0", p0[k, 0, 0])
                                    print("p1", p0[k, 0, 1])
                                    if (left-20 <= p0[k, 0, 0]) and \
                                            (p0[k, 0, 0] <= right+20) and \
                                            (top-20 <= p0[k, 0, 1]) and \
                                            (p0[k, 0, 1] <= bottom+20):
                                        label[k, 0, 0] = i
                        # print("label",label)
                        p0_con = np.concatenate((p0, label), axis=2)
                        p0_con_tmp = p0_con
                        p0_con = p0_con_tmp.reshape(-1, 1, 3)  #necessary
                        # print("p0_con", p0_con)
                        # print("p0", p0)
                        print("label", label)

                        # for i in range(p0_con.shape[0]):  # fold and enumerate with i
                        #     a = int(p0_con[i, 0, 0])
                        #     b = int(p0_con[i, 0, 1])
                        #     cv2.putText(drawimg, text=str(p0_con[i, 0, 2]), org=(a, b),
                        #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        #                 fontScale=1, color=(0, 255, 255), thickness=2)

                        # print("1p0.shape", p0_sigle.shape)
                        # print("1p0", p0_sigle)
                    # if len(target) == boxLenth:
                    #     for k in range(label.shape[0]):
                    #         if label[k, 0, 0] != -1:
                    #             flag = 1
                    #             break
                        print("unique", np.unique(label[label != -1]))
                        if (label != -1).any() and np.size(np.unique(label[label != -1])) > 1:
                            flag = 1
                        # if np.size(p0_con.shape[0]) > 0:
                        #     print("change!!!!!!!!!!!!!!!!!!!!!!!!!change!!!!!!!!!!!!!!!")
                        #     # target.append(p0_con[:, :, 0:2])
                        #     # print("len target:", len(target))
                        #     # print("target:", target)
                        #     # print("len target:", len(target))
                        #     # input = np.squeeze(p0_con[:, :, 0:2])
                        #     flag = 1
                        # else:
                        #     target = []
            else:
                # track
                if np.size(p0_con.shape[0]) > 0:
                    # print("len(target)", len(target))
                    # print("target", target)
                    # print("input", input)
                    # p0 = target[0]
                    # print("p0_chuan", p0)
                    # print("pk", pk)
                    # label = p0_con[:, :, 2:3]
                    # print("label", label)

                    p1, st, err = cv2.calcOpticalFlowPyrLK(featureimg, secondimg, p0, None, **lk_params)
                    if p1 is not None and (label != -1).any() and np.size(p1.shape[0]) > 0:
                        # print("st", st)
                        good_new = p1[st == 1]  # will error when twise  can not use the same
                        good_old = p0[st == 1]  # will error when twise

                        good_label = label[st == 1]
                        # print("good_label", good_label)
                        good_new_con = np.concatenate((good_new, good_label), axis=1)
                        good_old_con = np.concatenate((good_old, good_label), axis=1)
                        # print("good_new", good_new)
                        # print("good_old", good_old)
                        for i, (new, old) in enumerate(zip(good_new_con, good_old_con)):  # fold and enumerate with i
                            a, b, la = new.ravel()  # unfold
                            c, d, la = old.ravel()
                            # print("-" * 50)

                            a = int(a)
                            b = int(b)
                            c = int(c)
                            d = int(d)

                            if la != -1:
                                cv2.line(drawimg, (a, b), (c, d), (0, 255, 255), 1)
                                cv2.circle(drawimg, (a, b), 3, (0, 0, 255), -1)

                                cv2.putText(drawimg, text=str(la), org=(a, b),
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                            fontScale=1, color=(0, 255, 255), thickness=2)

                        p0 = good_new.reshape(-1, 1, 2)
                        label = good_label.reshape(-1, 1, 1)

                    """
                    for i in range(len(target)):
                        if target[i][0] is not None and target[i][2] == True:
                            p0 = target[i][0].copy()
                            print("flag ==2 " * 10)
                            if p1 is not None and st is not None:
                                good_new = p1[st == 1]  # will error when twise  can not use the same
                                good_old = p0[st == 1]  # will error when twise
                                dis = np.sum((good_new - good_old)**2, axis=1)
                                good_new = good_new[dis.argsort(axis=0)]
                                good_old = good_old[dis.argsort(axis=0)]
                                # len = good_new.shape[0]
                                if np.size(good_new) > 0:
                                    good_new = np.array([good_new[0]])
                                    good_old = np.array([good_old[0]])

                                    offset = good_new - good_old
                                    left +=offset[0, 0]
                                    top += offset[0, 1]
                                    right += offset[0, 0]
                                    bottom += offset[0, 1]

                                    posx = int(good_new[0, 0])
                                    posy = int(good_new[0, 1])
                                    print("posx", posx)
                                    print("posy", posy)
                                    idd = target[i][1]
                                    ID ="ID:" + str(idd)
                                    if posx > 1100:
                                        target[i][2] = False

                                    cv2.putText(drawimg, text=str(ID), org=(posx, posy),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                fontScale=1, color=(0, 255, 255), thickness=2)

                                    cv2.putText(drawimg, text=str(int(offset[0, 0])), org=(150, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                fontScale=1, color=(0, 255, 255), thickness=2)
                                    cv2.putText(drawimg, text=str(int(offset[0, 1])), org=(400, 35),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                fontScale=1, color=(0, 255, 255), thickness=2)
                                    print("i:", i)

                                    print("*" * 50)
                                    for i, (new, old) in enumerate(zip(good_new, good_old)):  # fold and enumerate with i
                                        a, b = new.ravel()  # unfold
                                        c, d = old.ravel()
                                        print("-" * 50)

                                        cv2.line(drawimg, (a, b), (c, d), (0, 255, 255), 1)
                                        cv2.circle(drawimg, (a, b), 3, (0, 0, 255), -1)

                                    p0 = good_new.reshape(-1, 1, 2)
                            target[i][0] = p0.copy()
                            """
            # k = False
            # for x in target:
            #     k = k or x[2]
            # if k is False:
            #     print("F FF clear the target" * 30)
            #     target = []
            #     flag = 0
            if "box" not in dataDict:
                # print("clear the target" * 30)
                target = []
                p0 = np.array([])
                p0_con = np.array([])
                label = np.array([])
                flag = 0
                cv2.circle(drawimg, (100, 100), 15, (0, 0, 255), -1)  # red  track
                left = 0
                top = 0
                right = 0
                bottom = 0
                                        # detect no bottle,back to detect

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
            #cv2.waitKey(1000)
            cv2.waitKey(10)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # return dataDict
            global bottleDict
            bottleDict = dataDict

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
    # _frame, nf = cam.getImage()
    print("准备载入yolo网络！")
    yolo = YOLO()
    print("准备背景学习！")
    bgobj = ImgProc(50)
    bgobj.studyBackgroundFromCam(cam)
    bgobj.createModelsfromStats(6.0)
    _image = Vision(cam, yolo, bgobj)
    print("开始！")
    global gState
    gState = 2
    return cam, _image


def imageRun(cam,_image):
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
    _image.detectSerialImage(cam, )
            # dataDict["bgTimeCost"] = _image.bgLearn.bgTimeCost
            #cv2.waitKey(10)
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

if __name__ == '__main__':
    cam, _image = imageInit()
    imageRun(cam, _image)
