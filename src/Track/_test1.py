from src.Vision.camera import Camera
# from src.Vision.vision import Image
# from src.Vision.yolo import *
import cv2
<<<<<<< HEAD:src/Track/_test1.py
# from src.Vision.imageProcess.imgProc import ImgProc
from src.Vision.imageProcess.bgLearn import Bglearn
=======
from src.Vision.imageProcess.imgProc import ImgProc
>>>>>>> e37de98efe0758be20affa29c5d30d0aad575bb4:src/Track/test1.py



if __name__ == '__main__':
    cam = Camera()
    while True:
        _frame, nFrame, t = cam.getImage()
        cv2.imshow("test", _frame)
<<<<<<< HEAD:src/Track/_test1.py
        tempImgproc = Bglearn(20)
        # tempImgproc = ImgProc(10)
=======
        tempImgproc = ImgProc(10)
>>>>>>> e37de98efe0758be20affa29c5d30d0aad575bb4:src/Track/test1.py
        frame, bgMask, resarray = tempImgproc.delBg(_frame) if tempImgproc else (_frame, None)
        dataDict = dict()
        #dataDict = ImageTrack.yolo.detectImage(img)
        #bgLearn = Bglearn()
        dataDict["nFrame"] = nFrame
        dataDict["frameTime"] = t  # 相机当前获取打当前帧nFrame的时间t
        dataDict["bgTimeCost"] = tempImgproc.bgTimeCost if tempImgproc else 0
        print(dataDict)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

<<<<<<< HEAD:src/Track/_test1.py
    cam.destroy()
=======
    cam.destroy()
>>>>>>> e37de98efe0758be20affa29c5d30d0aad575bb4:src/Track/test1.py
