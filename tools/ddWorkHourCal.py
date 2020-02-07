# coding=utf-8
import cv2
import numpy as np
from PIL import ImageGrab
import numpy
import time
from timeit import default_timer as timer
import xlsxwriter
import sys
from collections import Counter

from shutil import copyfile

# sys.setdefaultencoding('utf8')


def capture(left, top, right, bottom):
    # img = ImageGrab.grab(bbox=(left, top, right, bottom))
    img = ImageGrab.grab() # full screen
    img = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
    r, g, b = cv2.split(img)
    cv2.merge([b, g, r], img)
    return img

prev_time = timer()
prev_cap =capture(0, 0, 600, 400)
x = 2
working_threshold = 1000.0/2073600.0

#用画图量出左上角格子的左上和右下顶点坐标 填写在下面
leftop = (250, 60, 720, 320)
work_hours = 0
gw = leftop[2] - leftop[0]
gh = leftop[3] - leftop[1]

sudoku = {"John": [leftop[0], leftop[0]+gw, leftop[1], leftop[1]+gh, work_hours],
          "TaoTao": [leftop[0]+gw, leftop[0]+2*gw, leftop[1], leftop[1]+gh, work_hours],
          "LuChenyin": [leftop[0]+2*gw, leftop[0]+3*gw, leftop[1], leftop[1]+gh, work_hours],
          "FeiFei": [leftop[0], leftop[0]+gw, leftop[1]+gh, leftop[1]+2*gh, work_hours],
          "ZhiMing": [leftop[0]+gw, leftop[0]+2*gw, leftop[1]+gh, leftop[1]+2*gh, work_hours],
          "Hujie": [leftop[0]+2*gw, leftop[0]+3*gw, leftop[1]+gh, leftop[1]+2*gh, work_hours],
          "Tina": [leftop[0], leftop[0]+gw, leftop[1]+2*gh, leftop[1]+3*gh, work_hours],
          "DaPeng": [leftop[0]+gw, leftop[0]+2*gw, leftop[1]+2*gh, leftop[1]+3*gh, work_hours],
          "LouQiGe": [leftop[0]+2*gw, leftop[0]+3*gw, leftop[1]+2*gh, leftop[1]+3*gh, work_hours]
          }
# test = cv2.imread("E:\\EvolzedArmlogic\\BottleSort\\tools\\1.jpg")
# i = 0
# for key in sudoku.keys():
#     i += 1
#     print(key)
#     x = test[sudoku[key][2]:sudoku[key][3], sudoku[key][0]:sudoku[key][1]]
#     cv2.imshow("diff"+str(i), x)
# cv2.waitKey() #要抑制速度
# img = cv2.rectangle(self.show, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画矩形

workbook = xlsxwriter.Workbook("E:\\EvolzedArmlogic\\BottleSort\\tools\\工作时间.xlsx")
worksheet = workbook.add_worksheet('工作时间')
title = ['John', 'TaoTao', 'LuChenyin', 'FeiFei', 'ZhiMing', 'Hujie', 'Tina', 'DaPeng', 'LouQiGe']
worksheet.write_column('B1', title)

while 1:
    curr_time = timer()
    randomInterval = numpy.random.uniform(low=0.5 * x, high=2.0 * x, size=1)
    if curr_time - prev_time >= randomInterval:
        print("in")
        curr_cap = capture(0, 0, 600, 400)
        show = curr_cap.copy()
        diff0 = cv2.absdiff(curr_cap, prev_cap)
        diff0 = cv2.cvtColor(diff0, cv2.COLOR_BGR2GRAY)
        i = 0
        for key in sudoku.keys():
            i += 1
            print(key)
            diff = diff0[sudoku[key][2]:sudoku[key][3], sudoku[key][0]:sudoku[key][1]]
            # cv2.imshow("diff" + str(i), x)
            diffSize = diff[diff == 255].size
            totalSize = diff.size
            thresh = float(diffSize)/float(totalSize)
            print("diffSize", diffSize)
            print("thresh", thresh)
            print("area", diff.size)
            print(1000.0 / 2073600.0)
            if thresh > working_threshold:
            # if diffSize > 1000:
                sudoku[key][4] += round(randomInterval[0], 2)
                print(str(key)+" work_hours", sudoku[key][4])
                print(str(key)+"working!!!!")
        feature = []
        for key in sudoku.keys():
            cv2.putText(show, text=str(sudoku[key][4]), org=(sudoku[key][0]+10, sudoku[key][2]+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(0, 0, 255), thickness=2)
            feature.append(sudoku[key][4])
        worksheet.write_column('C1', feature)
        cv2.imshow("diff", show)
        cv2.waitKey(10) #要抑制速度
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # cam.destroy()
            workbook.close()

            break
        prev_time = curr_time
        prev_cap = curr_cap

rand = numpy.random.randint(1, 7, 1)
rand[0]=6
if rand[0] == 6:
    cv2.imwrite("E:\\EvolzedArmlogic\\BottleSort\\tools\\save.jpg", show)
    # cv2.waitKey(0)
