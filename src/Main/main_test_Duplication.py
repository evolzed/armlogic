# -- coding: utf-8 --
#!/bin/python
import os
import sys
sys.path.append(os.path.abspath('../../'))
from lib.Logger.Logger import *
import time
import logging
from src.Vision import vision

#logger = logging.getLogger(__name__)
#logger.addHandler(logging.FileHandler("log.txt"))  # 添一个FileHandler
#logging.basicConfig(filename="test.log",filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
#                    datefmt="%d-%M-%Y %H:%M:%S", level=logging.INFO)    #设置

def main():
    gState = vision.gState
    print("gState={},系统初始化···".format(gState))
    try:
        cam, _image = vision.imageInit()
    except Exception as e:
        print("系统初始化失败！--{}".format(e))
        sys.exit()
    else:
        print("系统初始化完成！")
        print("gState={}".format(vision.gState))

    if vision.gState == 2:
        print("gState={},系统启动中···".format(vision.gState))
        print("开始获取数据···")
        try:
            vision.imageRun(cam, _image)
        except Exception as e:
            print("系统启动失败！--{}".format(e))
            sys.exit()


#     global gState
#     global gDir
#
#     if gState == 1:
#
#         print("gState = 1")
#         time.sleep(1)
#         #logger.info("info")
#         #logger.error("error")
#         #logger.addHandler(log)
#         gState = 2
#     elif gState == 2:
#         print("gState = 2")
#         image.imageRun(cam, _image)
#         time.sleep(1)
#         #logger.error("error2")
#         gState = 3
#         #logger.addHandler(log)
#     elif gState == 3:
#         print("gState = 3")
#         time.sleep(1)
#         # if message == 1:
#         #     #logger.info("this is debug information ...")
#         #     gState = 1
#

if __name__ == "__main__":
    sys.stdout = Logger("D:\\log.txt")  # 保存到D盘
    #print("please type the gState: ")
    #gState = int(input())
    #print("please type the message (debug_test: 1 for bug ; 0 for normal): ")
    #message = int(input())
    # n = 0
    # while True:
    #     try:
    main()  #删除循环，让程序自主退出
        # except KeyboardInterrupt:
        #     sys.stderr.write("Keyboard interrupt.\n")
        #     sys.exit(main())
        # n += 1