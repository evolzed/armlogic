# -- coding: utf-8 --
#!/bin/python
import os
import sys
sys.path.append(os.path.abspath('../../'))
from lib.Logger.Logger import *
from src.Vision import vision


def main():
    gState = vision.gState   #gState 从Image获取
    print("gState={},系统初始化···".format(gState))
    try:
        cam, _image = vision.imageInit()     #相机初始化
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
            vision.imageRun(cam, _image)     #相机运行
        except Exception as e:
            print("系统启动失败！--{}".format(e))
            sys.exit()

if __name__ == "__main__":
    sys.stdout = Logger("D:\\log.txt")  # 保存到D盘
    main()
