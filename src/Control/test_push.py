#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2018, UFactory, Inc.
# All rights reserved.
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

import os
import sys
import time
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from uarm.wrapper import SwiftAPI

"""
api test: move
"""
#初始化机械臂状态；
swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'})
swift.waiting_ready(timeout=3)
device_info = swift.get_device_info()
print(device_info)
firmware_version = device_info['firmware_version']


swift.send_cmd_async('M2123 V0')#失步检测功能，V0为关闭，V1为开启；
swift.send_cmd_async('M2400 V5')#设置机械臂的末端模式，V5为末端步进电机；

time.sleep(1)

speed = 50
swift.set_acceleration(5)

Pos_X = 200
Pos_Y = 150
Pos_Z = 100

while True:
    swift.set_position(x=Pos_X, y=-Pos_Y, z=Pos_Z, speed=speed, cmd='G0')#cmd='G0'模式为弧线运动模式，G1为直线运动模式；
    #swift.set_servo_angle(3,0)
    swift.send_cmd_async('G2202 N3 V10 F100')#设置末端步进电机旋转角度，为绝对角度；
    swift.set_position(x=Pos_X, y=Pos_Y, z=Pos_Z, speed=speed, cmd='G0')
    #swift.set_servo_angle(3,120)
    swift.send_cmd_async('G2202 N3 V120 F100')
#
# time.sleep(1)
#
# while True:
#     if i == 1:
#         startTime = time.perf_counter()
#     if i > count:
#         break
#     swift.set_position(x=Pos_X, y=-Pos_Y, z=100, speed=speed, cmd='G0')
#     swift.flush_cmd()
#     swift.set_position(x=Pos_X, y=Pos_Y, z=100, speed=speed, cmd='G0')
#     swift.flush_cmd()
#     i = i + 1
#
# angle = 2 * math.atan(Pos_Y/Pos_X)/3.14*180
# radius = math.sqrt(Pos_Y*Pos_Y+Pos_X*Pos_X)
# L = angle * 3.14 * radius / 180
#
# endTime = time.perf_counter()
#
# Time = endTime - startTime
#
# print("往返次数：",count * 2)
# print("总耗时：",Time)
# print("弧长：",L)
# print("平均速度：",(i-1) * L / Time)
#
# swift.set_position(x=Pos_X, y=-Pos_Y, z=100, speed=speed, cmd='G0')


