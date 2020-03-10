# 本文件示例建立文件夹的代码
import sys
import os
#本地保存屏幕截图的路径
captureDir = "E:\\1\\Capture\\"
#本地存放钉钉录制视频的路径
videoDir = "E:\\1\\DDvedio\\ttt.mp4"
# videoDir = "E:\\1\\1.avi"
#本地存放背景视频的路径，这里没用，写和videoDir相同即可
bgDir = videoDir
#本地存放标定图片的路径
calibDir = "E:\\1\\Calib\\"
#Excel 存放路径
excelDir = "E:\\1\\WorkExcel\\"


if not os.path.exists(captureDir):
    os.mkdir(captureDir)
if not os.path.exists(excelDir):
    os.mkdir(excelDir)

if not os.path.exists(calibDir):
    os.mkdir(calibDir)

title = []
for i in len(range(10)):
    title.append(str(i))
print("title", title)

for i in range(len(title)):
    if not os.path.exists(captureDir + title[i]):
        os.mkdir(captureDir + title[i])
