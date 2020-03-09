import cv2
import time
from timeit import default_timer as timer

class Video:
    def __init__(self, videoDir):
        """

        :param videoDir:
        """
        self.cap = cv2.VideoCapture(videoDir)  # 获取视频对象
        self.frameTotalCount = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) #有效
        self.isOpened = self.cap.isOpened  # 判断是否打开
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.nFrame = 0
        self.time = timer()
        self.framCostTime = self.frameTotalCount/self.fps
        self.framInterval = 1 / self.fps

    def getCurrentTime(self):
        currentTime = self.framInterval* self.nFrame
        h = int(currentTime/3600)
        m = int(currentTime % 3600/60)
        s = int(currentTime % 3600 % 60)
        return currentTime, h, m, s


    def getFrameIdFromMoment(self, moment):
        frameId = moment / self.framCostTime * self.frameTotalCount
        frameId = int(frameId)
        return frameId


    def getImageFromVideoAtMoment(self, h, m, s):
        """

        :return:
        frame
        """
        if self.isOpened:
            #当前时刻
            moment = float(h*3600 + m*60 + s)
            #转换为当前对应的帧
            id = self.getFrameIdFromMoment(moment)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, id)
            (frameState, frame) = self.cap.read()  #记录每帧及获取状态
            if frameState == True:
                t = time.time()  # 获取当前帧的时间
                nF = id
                return frame, nF, t
            else:
                print("video over!!!!!!!")
                return None, None, None
        else:
            return None, None, None

    def resetVideo(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def setFrameID(self, idn):
        if idn <= self.frameTotalCount:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idn)
            self.nFrame = idn
            return 1
        else:
            return -1

    def getImageFromVideo(self):
        """

        :return:
        frame
        """
        if self.isOpened:
            (frameState, frame) = self.cap.read()  # 记录每帧及获取状态
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if frameState == True:
                t = time.time()  # 获取当前帧的时间
                self.nFrame += 1
                nF = self.nFrame
                return frame, nF, t
            else:
                print("video over!!!!!!!")
                return None, None, None
        else:
            return None, None, None

"""
视频播放的快,抽帧都是对的，帧和帧间隔时间是不对的，我们的间隔短，实际的间隔长
"""
if __name__ == '__main__':
    avi = Video("E:\\1\\DDvedio\\20200213-141916_胡杰_水印_video.mp4")
    frame = avi.getImageFromVideo()
    print("totalcount", avi.frameTotalCount)
    print("fps", avi.fps)
    print("framCostTime", avi.framCostTime)
    print("framInterval", avi.framInterval)
    frame = avi.getImageFromVideoAtMoment(2, 55, 56)
    cv2.imshow("test", frame)
    cv2.waitKey()

    avi.resetVideo()
    actual_cnt = 0
    while frame is not None:
        frame, nf, t = avi.getImageFromVideo()
        currentTime, h, m, s = avi.getCurrentTime()
        cv2.putText(frame, text=str(h) + "h",
                    org=(200, 800), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, text=str(m) + "m",
                    org=(260, 800), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, text=str(s) + "s",
                    org=(320, 800), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=2)
        actual_cnt += 1
        print("actual_cnt", actual_cnt)
        """
           Then write your frame process code here.
        """
        if frame is not None:
            cv2.imshow("avi", frame)
        cv2.waitKey(10)

""" 
def getImageFromVideo(VideoDir):
    cap = cv2.VideoCapture(VideoDir)  # 获取视频对象
    isOpened = cap.isOpened  # 判断是否打开
    # 视频信息获取
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    imageNum = 0
    while (isOpened):
        if imageNum / fps == 2:  # 获取两秒内视频图片
            break
        else:
            imageNum = imageNum + 1
        (frameState, frame) = cap.read()  # 记录每帧及获取状态
        fileName = 'image' + str(imageNum) + '.jpg'  # 存储路径

        if frameState == True:
            cv2.imwrite(fileName, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(fileName + "successfully write in")  # 输出存储状态
    print('finish!')
"""


