#import cv2
import numpy as np
#bgLearn
#Description:
#Learn the backgroud by pics from cam then get a background model
#bgLearn is implemented by sequential procedure, and  theses procedure are  expressed  several functions as below.
from lib.GrabVideo import GrabVideo
from lib.HikMvImport.CameraParams_header import MV_FRAME_OUT_INFO_EX
from ctypes import *
from timeit import default_timer as timer
import cv2


def studyBackgroundFromCam(cam): #get 100 pics for time interval of 60sec by cam and save the pics as background pics sets in disk.
    """
    Parameters
     --------------
     cam: input camera object


       Returns
    -------


       Examples
    --------
    """
    bgVector=[]
    BGDIR = "E:\\Xscx2019\\OPENCV_PROJ\\backgroundLearn\\"
    over_flag = 1
    BG_STUDY_NUM=10
    while (over_flag):
        pic_cnt = 1
        #Mat frame, blob;
        frame = cam.getImage()
        #Mat fin = Mat::zeros(frame.size(), CV_32FC3);
        fin=cv.zeros(frame.size(), CV_32FC3)
        print(np.array(pin))
        frame.convertTo(fin, CV_32FC3)
        bgVector.append(fin)
        picname = to_string(pic_cnt)
        picname = BGDIR + picname + ".jpg"
        print(picname)
        cv.waitKey(200)
        pic_cnt=pic_cnt+1
        if (pic_cnt == BG_STUDY_NUM):
            over_flag = 0

    #learnBackGroundFromVec(bgVector);

def avgBackground(img):#read background pics from disk,and then calculate every frame difference,and accumulate every frame difference
                   #to a sum of frame difference,and then calculate the average frame difference,meanwhile,accumulate every frame to a sum of frame and
                   #then calculate the average frame.
    """
    Parameters
    --------------
    img: input  Mat type pic stream


      Returns
   -------



      Examples
   --------
"""



def createModelsfromStats():  # average the frame and frame difference to get the background model
    """
    Parameters
    --------------
    I:      input cam pic waited for segment
    dst:    segment result




    Returns
    -------



    Examples
    --------
    """



def backgroundDiff(src,dst):# when get pic frame from camera, use the backgroundDiff to  segment the frame pic;
    # if the pic pixel value is higher than  high background threadhold  or lower than low background threadhold, the pixels
    # will change to white,otherwise, it will cover to black.

    """
        Parameters
        --------------
        I:      input cam pic waited for segment
        dst:    segment result




        Returns
        -------



        Examples
        --------
    """




#checkImage

#Description:
#checkImage is implemented by sequential procedure, and  theses procedure are  expressed  several functions as below.

#checkImage Implemente Details:


class Image(object):
    """create main Image class for processing images"""
    def detectVideo(self, yolo, output_path=""):
        """
        进行实时视频检测功能
        :param yolo:
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

