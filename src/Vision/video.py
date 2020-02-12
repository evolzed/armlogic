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

    def getImageFromVideo(self):
        """

        :return:
        frame
        """
        if self.isOpened:
            (frameState, frame) = self.cap.read()  # 记录每帧及获取状态
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
    avi = Video("E:\\1\\1.avi")
    frame = avi.getImageFromVideo()
    print("totalcount", avi.frameTotalCount)
    print("fps", avi.fps)
    print("framCostTime", avi.framCostTime)
    print("framInterval", avi.framInterval)

    actual_cnt = 0
    while frame is not None:
        frame, nf, t = avi.getImageFromVideo()
        actual_cnt += 1
        print("actual_cnt", actual_cnt)
        """
           Then write your frame process code here.
        """
        if frame is not None:
            cv2.imshow("avi", frame)
        cv2.waitKey(80)

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


