from src.Vision.camera import Camera
# from src.Vision.vision import Image
import cv2
from src.Vision.imageProcess.imageTrack import ImageTrack
from src.Vision.imageProcess.bgLearn import Bglearn


if __name__ == '__main__':
    cam = Camera()
    while True:
        _frame, nFrame, t = cam.getImage()
        cv2.imshow("test", _frame)

        frame, bgMask = Bglearn(10).delBg(_frame) if Bglearn(10) else (_frame, None)
        dataDict = dict()
        #dataDict = ImageTrack.yolo.detectImage(img)
        #bgLearn = Bglearn()
        dataDict["nFrame"] = nFrame
        dataDict["frameTime"] = t  # 相机当前获取打当前帧nFrame的时间t
        dataDict["bgTimeCost"] = Bglearn(10).bgTimeCost if Bglearn(10) else 0
        print(dataDict)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    cam.destroy()