import cv2

class Video:
    def __init__(self, videoDir):
        """

        :param videoDir:
        """
        self.cap = cv2.VideoCapture(videoDir)  # 获取视频对象
        self.isOpened = self.cap.isOpened  # 判断是否打开
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def getImageFromVedio(self):
        """

        :return:
        frame
        """
        if self.isOpened:
            (frameState, frame) = self.cap.read()  # 记录每帧及获取状态
            if frameState == True:
                return frame
            else:
                return None
        else:
            return None


if __name__ == '__main__':
    avi = Video("E:\\1\\1.avi")
    frame = avi.getImageFromVedio()
    while frame is not None:
        frame = avi.getImageFromVedio()
        """
           Then write your frame process code here.
        """
        cv2.imshow("avi", frame)
        cv2.waitKey(10)

""" 
def getImageFromVedio(vedioDir):
    cap = cv2.VideoCapture(vedioDir)  # 获取视频对象
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


