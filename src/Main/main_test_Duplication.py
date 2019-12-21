# -- coding: utf-8 --
#!/bin/python
import os
import sys
sys.path.append(os.path.abspath('../../'))
from lib.Logger.Logger import *
import time
import logging
from src.Image import image

#logger = logging.getLogger(__name__)
#logger.addHandler(logging.FileHandler("log.txt"))  # 添一个FileHandler
#logging.basicConfig(filename="test.log",filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
#                    datefmt="%d-%M-%Y %H:%M:%S", level=logging.INFO)    #设置

def main():
    gState = image.gState
    print("gState={},系统初始化···".format(gState))
    try:
        cam, _image = image.imageInit()
    except Exception as e:
        print("系统初始化失败！--{}".format(e))
        sys.exit()
    else:
        print("系统初始化完成！")
        print("gState={}".format(image.gState))

    if image.gState == 2:
        print("gState={},系统启动中···".format(image.gState))
        print("开始获取数据···")
        try:
            image.imageRun(cam, _image)
        except Exception as e:
            print("系统启动失败！--{}".format(e))
            sys.exit()


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