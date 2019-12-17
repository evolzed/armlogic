import os
import sys
import numpy as np
from ctypes import *
from timeit import default_timer as timer
import cv2
sys.path.append(os.path.abspath("../../"))
from lib.GrabVideo import GrabVideo
from lib.HikMvImport.utils.CameraParams_header import MV_FRAME_OUT_INFO_EX
from lib.camera import Camera
from yolo.Yolo import *


class Image(object):
    """create main Image class for processing images"""
    def __init__(self, cam):
        """相机自检"""
        self.cam = cam
        self.deviceNum = cam.get_device_num()
        self._cam, self._data_buf, self._nPayloadsize = self.cam.connect_cam(self.deviceNum)
        if -1 == self._cam:
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
        if self._cam == -1 or self._data_buf is None:
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
            image = PImage.fromarray(frame)
            image = yolo.detect_image(image)
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
            ret = self._cam.MV_CC_GetOneFrameTimeout(byref(self._data_buf), self._nPayloadsize, stFrameInfo, 1000)
            if ret == 0:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                    stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
            else:
                print("no data[0x%x]" % ret)
            if GrabVideo.g_bExit is True:
                break
        self.cam.destroy(self._cam, self._data_buf)
        yolo.close_session()


if __name__ == '__main__':
    cam = Camera()
    image = Image(cam)
    print("准备载入yolo网络！")
    yolo = YOLO()
    image.detectVideo(yolo)