# -- coding: utf-8 --
#!/bin/python
import os
import sys
# sys.path.append(os.path.abspath('../../'))
from lib.Logger.Logger import *
import time
import logging
from src.Image import image
gState = image.gState

sys.stdout = Logger("D:\\12.txt")  # 保存到D盘

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


# def main():
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
# if __name__ == "__main__":
#     sys.stdout = Logger("D:\\12.txt")  # 保存到D盘
#     # print("please type the gState: ")
#     # gState = int(input())
#     # print("please type the message (debug_test: 1 for bug ; 0 for normal): ")
#     # message = int(input())
#     n = 0
#     while True:
#         try:
#            main()
#         except Exception as e:
#            print(e)
#            sys.stderr.write("Keyboard interrupt.\n")
#            sys.exit(main())
#         n += 1