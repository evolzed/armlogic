#本文件示例不同显示接口的用法
# coding=utf-8
import cv2
import numpy as np
from PIL import ImageGrab
import numpy
import time
from timeit import default_timer as timer
from src.Vision.video import Video
from src.Vision.interface import imageCapture
import xlsxwriter
import sys
import os
from collections import Counter
from tools.timestamp import getTimeStamp

from shutil import copyfile
from tools.numeralRecognition import numRecogByMnistKnn
from tools.mnist import *
from tools.pyTorch import torchPred
from tools.pyTorch import Neural_net
from tools.pyTorch import findTheNumPic
videoDir = "E:\\1\\1.avi"
bgDir = videoDir
testMode = True

left = 0
top = 0
w = 100
h = 100
if __name__ =="__main__":

    avi = Video(videoDir)
    # print(avi.framInterval)
    bgAvi = Video(bgDir)
    imgCapObj = imageCapture(None, avi, bgAvi)
    # 重置视频到第0帧
    imgCapObj.resetCamFrameId()

    #初始化帧

    #当前帧的画面
    curr_cap0, nFrame0, t0 = imgCapObj.getImage()


    #显示
    show = curr_cap0.copy()
# b g r
    cv2.rectangle(show, (left, top),
                      (left + w, top + h), (0, 0, 255), 2)  # 画矩形

    txt = "k"

    #视频时间
    cv2.putText(show, text="video watch:",
                org=(400, 750), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 255, 255), thickness=2)


    cv2.putText(show, text=str(txt) + "h",
                org=(400, 800), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 255, 255), thickness=2)
    cv2.putText(show, text=str(txt) + "m",
                org=(480, 800), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 255, 255), thickness=2)
    cv2.putText(show, text=str(txt) + "s",
                org=(560, 800), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 255, 255), thickness=2)

    cv2.namedWindow("window", 0)
    cv2.resizeWindow("window", 1920, 1080)
    cv2.imshow("window", show)

    # cv2.imwrite(captureDir + str(key)+"\\"+str(time_stamp)+".jpg", show0.copy())
    cv2.waitKey()

