#本文件示例不同时间的帧的处理框架
from src.Vision.video import Video
from src.Vision.interface import imageCapture
from timeit import default_timer as timer
import cv2
import numpy as np

videoDir = "E:\\1\\1.avi"
bgDir = videoDir
testMode = True
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
    #前帧的画面
    prev_cap = curr_cap0.copy()
    #前帧的帧号
    preNframe = nFrame0
    #前帧的时间
    prev_time = timer()

    #要隔的间隔
    frameInterval = imgCapObj.getCamFrameInterval()

    #显示
    show = curr_cap0.copy()
    cv2.namedWindow("window", 0)
    cv2.resizeWindow("window", 1920, 1080)

    while 1:
        if testMode:
            curr_cap, nFrame, t = imgCapObj.getImageFromCamAtMoment(0, 0, 1)
        else:
            curr_cap, nFrame, t = imgCapObj.getImage()
        #当前帧的拷贝作为显示
        show = curr_cap.copy()
        curr_time = timer()
        # 间隔多久时间进行一次视频比对
        randomInterval = 20
        diff0 = cv2.absdiff(curr_cap, prev_cap)
        if frameInterval * (nFrame - preNframe) >= randomInterval:
            if 0:
                pass  #process

            # update the time and frame every time up
            #更新前帧时间
            prev_time = curr_time
            # 更新前帧画面
            prev_cap = curr_cap
            # 更新前帧帧号
            preNframe = nFrame

        cv2.imshow("window", show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # 按q退出 并保存excel
            break