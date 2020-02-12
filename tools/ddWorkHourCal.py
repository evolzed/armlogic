# coding=utf-8
import cv2
import numpy as np
from PIL import ImageGrab
import numpy
import time
from timeit import default_timer as timer
from src.Vision.video import Video
from src.Vision.interface import imageCapture
import xlsxwriter
import sys
import os
from collections import Counter

from shutil import copyfile


def capture(left, top, right, bottom):
    """
    :param left: 想要截取图像的最左侧坐标
    :param top: 想要截取图像的最顶侧坐标
    :param right: 想要截取图像的最右侧坐标
    :param bottom: 想要截取图像的最下侧坐标
    :return: img  返回截取到的图像
    """
    # img = ImageGrab.grab(bbox=(left, top, right, bottom))
    img = ImageGrab.grab() # full screen
    img = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
    r, g, b = cv2.split(img)
    cv2.merge([b, g, r], img)
    return img


# 本工具运行方法，首先设定本地保存屏幕截图的路径captureDir，设定存放钉钉录制视频的路径videoDir，
# 设定保存标定图片的路径calibDir，设定保存工作时间表的路径excelDir，
# 然后运行程序，在标定图片路径下获得了标定图片，
# 用画图软件打开这张图片，看左上角点的坐标，填写在left,top上，看一个格子的宽度和高度，填写在
# gw，gh上,根据钉钉视频上九宫格每个人的位置，修改sudoku字典人名的位置，每个人途中不得离开，
# 可以关闭摄像头，否则位置会变化
# 关闭程序
# 再重新运行程序，投掷骰子，如果骰子掷为6，则随机时间保存屏幕截图，否则不保存，观察运行窗口，
# 实时查看每个人的工作时间显示在左上角，小数位数较多，每分钟会更新一次
# 工作结束按q退出,会保存每个人的工作时间在excel表中，excel表的文件名已经加上时间戳


#本地保存屏幕截图的路径
captureDir = "E:\\1\\Capture\\"
#本地存放钉钉录制视频的路径
videoDir = "E:\\1\\DDvedio\\20200211-130438_afternoon.mp4"
# videoDir = "E:\\1\\1.avi"
#本地存放背景视频的路径，这里没用，写和videoDir相同即可
bgDir = videoDir
#本地存放标定图片的路径
calibDir = "E:\\1\\Calib\\"
#Excel 存放路径
excelDir = "E:\\1\\WorkExcel\\"

# 九宫格子的起点坐标 和格子的宽度 和 高度，根据自己电脑显示屏 视频的窗口大小来填写，具体方法是取视频中的一帧
# 保存为一张图片，该程序跑起来后会保存为一张名字叫做 标定.jpg的图片，
# 用画图软件打开这张图片，看左上角点的坐标，填写在left,top上，看一个格子的宽度和高度，填写在
# gw，gh上
left = 0
top = 0
gw = 637
gh = 360

work_hours = 0
#九宫格的位置，每次需要根据实际的位置修改下列人名的位置
sudoku = {
          "Hujie":      [left,       left+gw,    top,       top+gh,    work_hours],
          "TaoTao":  [left+gw,    left+2*gw,  top,       top+gh,    work_hours],
          "LuChenYin":     [left+2*gw,  left+3*gw,  top,       top+gh,    work_hours],

          "John":       [left,       left+gw,    top+gh,    top+2*gh,  work_hours],
          "FeiFei":     [left+gw,    left+2*gw,  top+gh,    top+2*gh,  work_hours],
          "Tina":       [left+2*gw,  left+3*gw,  top+gh,    top+2*gh,  work_hours],

          "LouQiGe":    [left,       left+gw,    top+2*gh,  top+3*gh,  work_hours],
          "DaPeng":    [left+gw,    left+2*gw,  top+2*gh,  top+3*gh,  work_hours],
          "ZhiMing":     [left+2*gw,  left+3*gw,  top+2*gh,  top+3*gh,  work_hours]
          }
if __name__ == '__main__':
    # 调试参数  1分钟多少秒 正常运行情况下60  调试情况下可改小 需要加大时间间隔可以改大
    seconds = 60
    # 每个员工屏幕格子是否改变的阈值 以比值来写 适用于不同电脑的显示屏
    working_threshold = 500.0/(gw*gh)
    #初始化工作时间


    # 这里要按照上面sudoku的顺序填写
    title = []
    for key in sudoku.keys():
        title.append(str(key))
    print("title", title)

    # 建立存放截图的文件夹
    if not os.path.exists(captureDir):
        os.mkdir(captureDir)
    if not os.path.exists(excelDir):
        os.mkdir(excelDir)

    if not os.path.exists(calibDir):
        os.mkdir(calibDir)

    for i in range(len(title)):
        if not os.path.exists(captureDir + title[i]):
            os.mkdir(captureDir + title[i])

    #存放工作时间的excel表
    #时间戳
    time_day = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    workbook = xlsxwriter.Workbook(excelDir + str(time_day) + "工作时间.xlsx")
    worksheet = workbook.add_worksheet('工作时间')

    worksheet.write_column('B1', title)

    #构造视频输入类 将钉钉视频会议录制的视频路径写在videoDir中
    avi = Video(videoDir)
    # print(avi.framInterval)
    bgAvi = Video(bgDir)
    imgCapObj = imageCapture(None, avi, bgAvi)
    # print(imgCapObj.video.framInterval)
    # 初始化图片用于初始化
    curr_cap0, nFrame0, t0 = imgCapObj.getImage()
    prev_cap = curr_cap0.copy()

    frameInterval = imgCapObj.getCamFrameInterval()

    # 保存标定图片路径，记得填写正确的路径 路径不能有中文 OPENCV对中文目录支持不好
    cv2.imwrite(calibDir + "biaoding.jpg", prev_cap)

    diff0 = curr_cap0.copy()
    show = curr_cap0.copy()
    prev_time = timer()

    # 投掷骰子来决定是否保存画面
    save_flag = False
    inputstr = input("投骰子决定是否 随机间隔保存截屏,投中6则保存，否则不会保存，请按任意键。。。")
    rand = numpy.random.randint(1, 7, 1)
    print("Your roll is ", rand[0])
    # rand[0] = 6
    if rand[0] == 6:
        print("You rolled 6,good luck! Then everyone capture of random time will be saved!")
        save_flag = True
    else:
        save_flag = False

    cv2.namedWindow("window", 0)

    cv2.resizeWindow("window", 1920, 1080)
    while 1:
        curr_cap, nFrame, t = imgCapObj.getImage()
        if curr_cap is None:
            break
        show = curr_cap.copy()
        curr_time = timer()
        # 间隔多久时间进行一次视频比对
        randomInterval = numpy.random.uniform(low=1.0 * seconds, high=2.0 * seconds, size=1)
        if curr_time - prev_time >= randomInterval:
            print("in")
            # show0 = np.zeros_like(curr_cap)
            # show0 = show0[1:100, 1:100].copy()
            # 计算设定时间间隔内视频的差值
            diff0 = cv2.absdiff(curr_cap, prev_cap)
            diff0 = cv2.cvtColor(diff0, cv2.COLOR_BGR2GRAY)
            ret, diff0 = cv2.threshold(diff0, 100, 255, 1)
            i = 0
            # 检索每个员工格子的视频差值 分析 得出每个员工的工作时间
            for key in sudoku.keys():
                i += 1
                print(key)
                diff = diff0[sudoku[key][2]:sudoku[key][3], sudoku[key][0]:sudoku[key][1]]
                show0 = show[sudoku[key][2]:sudoku[key][3], sudoku[key][0]:sudoku[key][1]]
                # 保存截图
                if save_flag:
                    time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
                    cv2.imwrite(captureDir + str(key)+"\\"+str(time_stamp)+".jpg", show0.copy())
                    # cv2.imwrite("E:\\1\\屏幕截图\\FeiFei\\563.766032.jpg", curr_cap.copy())
                    # print(captureDir+str(key)+"\\"+str(curr_time)+".jpg")

                # 统计每个员工格子的差值像素的数量 超出阈值 则累计工作时间
                diffSize = diff[diff == 0].size
                totalSize = diff.size
                thresh = float(diffSize)/float(totalSize)
                print("diffSize", diffSize)
                print("thresh", thresh)
                print("area", diff.size)
                if thresh > working_threshold:
                # if diffSize > 1000:
                    #因为抽帧快，不等于现实时间，所以增加原视频的帧间隔时间来补偿
                    sudoku[key][4] += (randomInterval[0]+ frameInterval - 10/1000)/3600.0
                    print(str(key)+" work_hours", sudoku[key][4])
                    print(str(key)+"working!!!!")
            feature = []
            for key in sudoku.keys():
                feature.append(sudoku[key][4])
            worksheet.write_column('C1', feature)
            #update the time and frame every time up
            prev_time = curr_time
            prev_cap = curr_cap
        # 实时显示格子和每个员工的工作时间和每个员工的名字，方便对照格子是否正确
        for key in sudoku.keys():
            cv2.putText(show, text=str(int(sudoku[key][4]*60/60))+" h", org=(sudoku[key][0]+30, sudoku[key][2]+30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=2)
            cv2.putText(show, text=str(int(sudoku[key][4]*60 % 60)) + " m", org=(sudoku[key][0] + 100, sudoku[key][2] + 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 0, 255), thickness=2)
            #人名
            cv2.putText(show, text=str(key), org=(sudoku[key][0] + 90, sudoku[key][2] + 90),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 0, 255), thickness=2)
            cv2.rectangle(show, (sudoku[key][0], sudoku[key][2]), (sudoku[key][0] + gw, sudoku[key][2] + gh),
                          (0, 255, 255))

        cv2.imshow("window", show)

        cv2.waitKey(10)  # 要抑制速度
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # 按q退出 并保存excel

            break
    #store the excel
    workbook.close()
