# -- coding: utf-8 --
#!/bin/python
import os
import sys
sys.path.append(os.path.abspath('../../'))
from lib.Logger.Logger import *
from src.Image import image


def main():
    gState = image.gState   #gState 从Image获取
    print("gState={},系统初始化···".format(gState))
    try:
        cam, _image = image.imageInit()     #相机初始化
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
            image.imageRun(cam, _image)     #相机运行
        except Exception as e:
            print("系统启动失败！--{}".format(e))
            sys.exit()

if __name__ == "__main__":
    sys.stdout = Logger("D:\\log.txt")  # 保存到D盘
    main()
