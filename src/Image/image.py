import os
import sys
import numpy as np
from ctypes import *
from timeit import default_timer as timer
import cv2
sys.path.append(os.path.abspath("../../"))
# from lib.GrabVideo import GrabVideo
from lib.HikMvImport.utils.CameraParams_header import MV_FRAME_OUT_INFO_EX
from camera import Camera, g_bExit
from yolo.Yolo import *
from src.Image.imageProcess.bgLearn import Bglearn


class Image(object):
    """create main Image class for processing images"""
    def __init__(self, cam, yolo, bgLearn):
        """相机自检"""
        self.cam = cam
        self.yolo = yolo
        self.bgLearn=bgLearn
        self.deviceNum = cam.getDeviceNum()
        self._data_buf, self._nPayloadsize = self.cam.connectCam()
        if -1 == self._data_buf:
            print("相机初始化失败！退出程序！")
            sys.exit()
        print("相机初始化完成！")

    def detectVideo(self, yolo, output_path=""):
        """
        进行实时视频检测功能
        :param yolo:yolo实例对象
        :return:
        """
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        if self._data_buf == -1 or self._data_buf is None:
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
            # print(self._data_buf)
            frame = np.asarray(self._data_buf)
            frame = frame.reshape((960, 1280, 3))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = PImage.fromarray(frame)  # PImage: from PIL import Image as PImage
            # image.show()
            image = yolo.detectImage(image)

            result = np.asarray(image)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", result)
            if isOutput:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret = self.cam.MV_CC_GetOneFrameTimeout(byref(self._data_buf), self._nPayloadsize, stFrameInfo, 1000)
            if ret == 0:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                    stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
            else:
                print("no data[0x%x]" % ret)
            if g_bExit is True:
                break
        self.cam.destroy(self.cam, self._data_buf)
        yolo.closeSession()

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
        img = PImage.fromarray(frame)  # PImage: from PIL import Image as PImage
        # img.show()
        # feed data into model
        dataDict = self.yolo.detectImage(img)
        curr_time = timer()
        exec_time = curr_time - prev_time
        dataDict["timecost"] = exec_time
        dataDict["nFrame"] = nFrame
        arr = np.asarray(dataDict["image"])
        cv2.imshow("ff", arr)
        #cv2.waitKey(1000)
        cv2.waitKey(10)
        return dataDict

"""
if __name__ == '__main__':
    cam = Camera()
    _frame, nf = cam.getImage()
    print("准备载入yolo网络！")
    yolo = YOLO()

   
    _image = Image(cam, yolo)
    dataDict = _image.detectSingleImage(_frame, nf)
    print(dataDict)
    # image.detectVideo(yolo)
"""

if __name__ == '__main__':
    cam = Camera()
    _frame, nf = cam.getImage()
    print("准备载入yolo网络！")
    yolo = YOLO()
    print("准备背景学习！")
    bgobj = Bglearn()
    bgobj.studyBackgroundFromCam(cam)
    bgobj.createModelsfromStats(6.0)
    _image = Image(cam, yolo, bgobj)
    print("开始！")
    while 1:
        try:
            _frame, nf = cam.getImage()
            frameDelBg = _image.bgLearn.delBg(_frame)
            dataDict = _image.detectSingleImage(frameDelBg, nf)
            #cv2.waitKey(10)
            print(dataDict)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(e)
            break
    cam.destroy()
