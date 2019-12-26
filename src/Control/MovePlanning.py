#!/usr/bin/env python3
# Author: LouYufeng <502550265@qq.com>


import sys
import os
import numpy as np
import operator

sys.path.insert(0, os.path.abspath("../../"))


# from src.Control.Px2W import *
from src.Control.Px2W import Zc, IntrinsicMtx, ExtrinsicMtx, Px2World
from lib.uArmSDK.uarm.wrapper import SwiftAPI

#机械臂初始化
swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'}, enable_handle_thread=False)
# swift.waiting_ready() #等待手臂完成初始化
# swift.send_cmd_async('M2400 S0')  # 设置机械臂工作模式，为S0常规模式
# swift.send_cmd_async('M2123 V1')  # 开启失步检测功能
# swift.set_digital_direction(pin = 32,value=1) #设置30Pin数字口D25为输出
# speed =400  # 弧线运动的速度，500
# swift.set_acceleration(15)  # 设置加速度，20


# BottleType, u, v, angle2Xw,Xw, Yw, Zw, distance2uArm 模拟矩阵
#     0       1  2      3      4   5   6        7
bottleDict = {"box": [["Bottle1", 100, 900, 30],
                      ["Bottle3", 300, 800, 20],
                      ["Bottle2", 600, 700, 50],
                      ["Bottle1", 900, 600, 60],
                      ["Bottle2", 700, 500, 10],
                      ["Bottle1", 200, 600, 15],
                      ["Bottle3", 100, 800, 25],
                      ["Bottle1", 800, 900, 35]], "ABC1": [1, 2, 3], "ABC2": [1, 2, 3]}

uArmPos = [500, 600, 10]  # 在世界坐标系下，uArm的位置
Px = bottleDict["box"]  # 瓶子信息矩阵
numRow = np.shape(Px)  # 瓶子数目

#  从像素坐标计算到世界坐标系下的坐标，并放入瓶子信息矩阵
for i in range(0, numRow[0]):
    worldPos = Px2World(Px[i][1], Px[i][2], Zc, IntrinsicMtx, ExtrinsicMtx)
    distance = float(np.sqrt((worldPos[0]-uArmPos[0])**2+(worldPos[1]-uArmPos[1])**2))
    for j in range(0, 3):
        Px[i].append(float(worldPos[0]))
    Px[i].append(distance)
    Px[i].append(isDone)

Px.sort(key=operator.itemgetter(7), reverse=False) # 对矩阵的第八列距离值进行升序排列，距离最近的先取

# 吸，调姿态，放,电磁继电器的CH3可以正常使用，其他可能会有问题
swift.set_wrist(angle=90)
for i in range(0, numRow[0]):
    swift.set_position(Px[i][4], Px[i][5], Px[i][6], speed=speed, wait=True, cmd='G0')
    swift.set_digital_output(pin=32, value=1, wait=True, timeout=10)
    swift.set_position(Px[i][4], Px[i][5], Px[i][6] + 50, speed=speed, wait=True, cmd='G0')
    bottomAngle0 = swift.get_servo_angle(servo_id=0)
    swift.set_position(50, Px[i][5], Px[i][6] + 50, speed=speed, wait=True, cmd='G0')
    bottomAngle1 = swift.get_servo_angle(servo_id=0)
    deltaBottomangle = bottomAngle1 - bottomAngle0
    swift.set_wrist(angle=90 + (Px[i][3] + deltaBottomangle))
    swift.set_digital_output(pin=32, value=0, wait=True, timeout=10)



