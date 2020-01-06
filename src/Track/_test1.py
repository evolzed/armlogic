from src.Vision.camera import Camera
# from src.Vision.vision import Image
# from src.Vision.yolo import *
import cv2
from PIL import Image as PImage
# from src.Vision.imageProcess.imgProc import ImgProc
from src.Vision.imageProcess.imgProc import ImgProc
from src.Track.Track import Track


# 
if __name__ == '__main__':
    cam = Camera()




    # test
    bottledict = {'target': [["f025d3fe-2b6e-11ea-a086-985fd3d62bfb", 0, [300, 300], [10, 10], 0, 0, 0],
                              ["kkkkkkkk-2b6e-11ea-a086-985fd3d62bfb", 0, [400, 400], [10, 10], 0, 0, 0]],
                  'bgTimeCost': 0.10440749999999888, 'timeCost': 1578021153.380255, 'nFrame': 0, 'frameTime': 0, 'targetTrackTime':0}
    tempdict = bottledict

    while True:
        _frame, nFrame, t = cam.getImage()
        # tempImgproc = ImgProc(10)
        # img = PImage.fromarray(_frame)
        # tempPos = list()
        # tempPos .append(bottledict.get("target")[0][2])
        # i = 0
        # tempPos[i + 1][0] = tempPos[i][0]
        tempdict["nFrame"] = nFrame
        tempdict["frameTime"] = t
        # tempdict = Track().updateTarget(tempdict)

        # print(tempdict)
        # print(tempdict)
        # cv2.line(_frame, (0, 0), (512, 512), (0, 0, 255), 1)
        # print(t - tempdict["targetTrackTime"])
        if (tempdict["targetTrackTime"] == 0 or abs(t - tempdict["targetTrackTime"]) < 0.2):
            tempdict = Track().updateTarget(tempdict)
            [a, b] = tempdict["target"][0][2]
            cv2.rectangle(_frame, (int(a), int(b)), (int(a) + 20, int(b) + 20), (125, 0, 125), 4)
        cv2.imshow("test", _frame)
        tempImgproc = ImgProc(10)

        frame, bgMask, resarray = tempImgproc.delBg(_frame) if tempImgproc else (_frame, None)
        # dataDict = bottledict
        #dataDict = ImageTrack.yolo.detectImage(img)
        #bgLearn = Bglearn()
        # dataDict["nFrame"] = nFrame
        # dataDict["frameTime"] = t  # 相机当前获取打当前帧nFrame的时间t
        # dataDict["bgTimeCost"] = tempImgproc.bgTimeCost if tempImgproc else 0
        # print(dataDict)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.destroy()