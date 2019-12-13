#main.py,主要是调用所有功能模块，主要分为三部分:
        第一部分为设备检查部分，gState = 1;
        第二部分是正常交互模式（是后期用户交互使用），gState = 2;
        第三部分是Debug模式（为后面调试阶段做准备），gState = 3
        
#!/usr/bin/python
#导入所需工作包
import sys      				#sys模块是python自带系统模块，可通过此模块方法进行对localhost的cmd操控。
import os       				#os模块是路径模块，通常用os来完成路径文件的调用和查看，一般和sys合用。
import Image                                    #导入图像处理的功能包
import Tool                                     #导入工具集功能包


def main()
    global gState
    global gDir
    if gState == 1:
        Tool.mySQL().connectDict()
        Tool.mySQL().createDict()
        Image.cameraOn()
        Image.getImage()
        Image.bgLearn()
        Tool.mySQL().updateDict(bgDict)
        print("gState = 1")
        gState = 2
    elif gState == 2:
        Image.getImage()
        Image.checkImage()
        Tool.mySQL().updateDict(bottledict)
        print("gState = 2")
    elif gState == 3:
        Tool.mySQL.disconnectDict()
        print("gState = 3")
    
if __name__ == "__main__":
    print("please type the gState: ")
    gState = int(input())
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("Keyboard interrupt.\n")
        sys.exit(main())
