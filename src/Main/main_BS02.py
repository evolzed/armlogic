# -- coding: utf-8 --
#!/bin/python
import os
import sys

# from src.BS02.track import track
from src.BS02.track.track import vision_run, Track, np, video_run
# from src.BS02.track.track_hjtest import vision_run, Track, np, video_run
sys.path.append(os.path.abspath('../../'))
from lib.Logger.Logger import *
from src.BS02 import vision
import multiprocessing

def main():



    # gState = BS02.gState   #gState 从Image获取
    # print("gState={},系统初始化···".format(gState))
    # try:
    #     cam, _image = BS02.imageInit()     #相机初始化
    # except Exception as e:
    #     print("系统初始化失败！--{}".format(e))
    #     sys.exit()
    # else:
    #     print("系统初始化完成！")
    #     print("gState={}".format(BS02.gState))
    #
    # if BS02.gState == 2:
    #     print("gState={},系统启动中···".format(BS02.gState))
    #     print("开始获取数据···")
        track = Track()

        with multiprocessing.Manager() as MG:  # 重命名

            transDict = MG.dict()
            transList = MG.list()
            targetDict = MG.dict()
            Flag = MG.list()
            if len(Flag) == 0:
                Flag.append(0)

            transFrame = multiprocessing.RawArray('d', np.zeros((6, 7, 3), np.double).ravel())
            # first line code is without filter; second line one is with filter
            p2 = multiprocessing.Process(target=track.trackProcess, args=(transDict, transList, targetDict, Flag))
            # p2.daemon = True
            p2.start()

            feed = int(input("please choose the feed (camera = 0 or video =1): "))
            if feed == 0:
                p1 = multiprocessing.Process(target=vision_run, args=(transDict, transList, targetDict, transFrame, Flag))
                # p1.daemon = True
                p1.start()
                p1.join()
            else:
                p1 = multiprocessing.Process(target=video_run, args=(transDict, transList, targetDict, transFrame, Flag))
                # p1.daemon = True
                p1.start()
                p1.join()

        # BS02.imageRun(cam, _image, transDict, transList, targetDict, transFrame)     #相机运行
        # try:
        #     BS02.imageRun(cam, _image)     #相机运行
        # except Exception as e:
        #     print("系统启动失败！--{}".format(e))
        #     sys.exit()


if __name__ == "__main__":
    # sys.stdout = Logger("D:\\log.txt")  # 保存到D盘
    main()
