
import numpy as np

from lib.GrabVideo import GrabVideo
from lib.HikMvImport.CameraParams_header import MV_FRAME_OUT_INFO_EX
from ctypes import *
from timeit import default_timer as timer
import cv2



class Image(object):
    """create main Image class for processing images"""
    def __init__(self):
        """相机自检"""

    def detectVideo(self, yolo, output_path=""):
        """
        进行实时视频检测功能
        :param yolo:yolo实例对象
        :return:
        """

        device_num = GrabVideo.get_device_num()
        cam, data_buf, nPayloadsize = GrabVideo.connect_cam(device_num)
        # print(data_buf)
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        if cam is None or data_buf is None:
            raise IOError("Couldn't open webcam or video")
        # vid = cv2.VideoCapture(video_path)
        # while True:
        #     temp = np.asarray(data_buf)
        #     temp = temp.reshape((960, 1280, 3))
        #     # print(temp)
        #     # print(temp.shape)
        #     temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        #     cv2.namedWindow("ytt", cv2.WINDOW_AUTOSIZE)
        #     cv2.imshow("ytt", temp)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # GrabVideo.destroy(cam, data_buf)
        # video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
        # 视频编码格式
        video_FourCC = 6
        # fps
        # video_fps       = vid.get(cv2.CAP_PROP_FPS)
        video_fps = 30
        # video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
        #                     int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
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
            print(data_buf)
            temp = np.asarray(data_buf)
            temp = temp.reshape((960, 1280, 3))
            # print(temp)
            # print(temp.shape)
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            # cv2.namedWindow("ytt", cv2.WINDOW_AUTOSIZE)
            # cv2.imshow("ytt", temp)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # return_value, frame = vid.read()
            frame = temp
            image = Image.fromarray(frame)
            image = yolo.detect_image(image)
            result = np.asarray(image)
            # print(type(result))
            # print(result)
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
            # result = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", result)
            if isOutput:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret = cam.MV_CC_GetOneFrameTimeout(byref(data_buf), nPayloadsize, stFrameInfo, 1000)
            if ret == 0:
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                    stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
                # print(stFrameInfo.nChunkHeight, stFrameInfo.nChunkWidth)
            else:
                print("no data[0x%x]" % ret)
            if GrabVideo.g_bExit is True:
                break
        GrabVideo.destroy(cam, data_buf)
        yolo.close_session()
