#main.py,主要是调用所有功能模块，主要分为三部分:
#        第一部分为设备检查部分，gState = 1;
#        第二部分是正常交互模式（是后期用户交互使用），gState = 2;
#        第三部分是Debug模式（为后面调试阶段做准备），gState = 3
        
#!/usr/bin/python
#导入所需工作包
import sys      				#sys模块是python自带系统模块，可通过此模块方法进行对localhost的cmd操控。
import os       				#os模块是路径模块，通常用os来完成路径文件的调用和查看，一般和sys合用。
import Image                                    #导入图像处理的功能包

def main():
    global gState                               #全局变量gState；
    #global gDir                                 #全局变量gDir；
    if gState == 1:                             #判断gState状态；若为 1 ，进入init，设备检查部分；
        Image.connectCam()                      #连接相机
        Image.loadYolo()                        #Tiny-Yolov3模型参数初始化;
        Image.bgLearn()                         #learn the backgroud by pics from cam then get a background model；
        Image.checkState()                      #检测gState状态，返回 1 ；
        print("gState = 1")                     #打印gState状态；
        gState = 2                              #设置给State状态为 2 ；
    elif gState == 2:                           #判断gState状态；若为 2 ，进入run，正常交互状态；
        Image.generate()                        #
        Image.detectImage()                    #检测输入图像
        Image.checkImage()                      #check the cnn detected result by image process and image track and then update the bottle dict
        Image.checkState()                      #检测gState状态，返回 2 ；
        print("gState = 2")                     #打印gState状态；
    elif gState == 3:                           #判断gState状态；若为 3 ，进入debug，调试阶段；
        print("gState = 3")                     #打印gState状态；
        #   if ()...
        gState = 1
if __name__ == "__main__":                      #main模块
    print("please type the gState: ")
    gState = int(input())                       #手动输入gState
    while (True):
        try:                                                    # 执行main()
            main()
        except KeyboardInterrupt:
            sys.stderr.write("Keyboard interrupt.\n")
            sys.exit(main())
