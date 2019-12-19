#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2018, UFactory, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

import os
import sys
import time
import math
import random

import functools
import threading
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from lib.uArmSDK.uarm.wrapper import SwiftAPI

swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'}, enable_handle_thread=False)
# swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'}, enable_write_thread=True)
# swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'}, enable_report_thread=True)
# swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'}, enable_handle_thread=True, enable_write_thread=True, enable_report_thread=True)
swift.waiting_ready() #等待手臂完成初始化

print(swift.get_device_info()) #机械臂固件4.3.2版本为高性能模式，如需要使用激光雕刻/3D打印等功能，请将固件降为3.2.0版本，升级/降级教程参考：https://pan.baidu.com/s/16F-tPocukHLSP4yHc-L1Mw 提取码 394p
swift.send_cmd_async('M2400 S0')  # 设置机械臂工作模式，为S0常规模式
swift.send_cmd_async('M2123 V1')  # 开启失步检测功能

resetSpeed = 50 #重置位置时，机械臂的运动速度
speed = 100   #弧线运动的速度，500
swift.set_acceleration(5)  #设置加速度，20

def Reset():
    swift.set_position(150,0,100,speed = resetSpeed,wait=False,timeout=10,cmd='G0')#G0模式为两点之间以最快方式到达的模式，G1模式为两点之间以直线到达的模式
    swift.flush_cmd()#清除缓存，保证前序指令一定执行完后，再执行下一个指令。
    time.sleep(1) #延时1秒

def Move(x,y,z,speed):
    swift.set_position(x=x,y=y,z=z,speed=speed,wait=False,timeout=10,cmd='G0')
    swift.flush_cmd()

def Suction_On():
    swift.set_pump(True)
    time.sleep(0.5)
    swift.flush_cmd()

def Suction_Off():
    swift.set_pump(False)
    time.sleep(0.5)
    swift.flush_cmd()
Reset()


#锯齿+门轨迹
# speed =100  #最大值400
# swift.set_acceleration(5)  #最大值20
# for y in range(-100, 100, 50):   #取离散的y坐标值
#     Move(140, y, 20, speed)  #运动到(x,y,z,速度)
#     Move(140, y, 100, speed)
#     Move(220, y, 100, speed)
#     Move(220, y, 20, speed)

#吸，调姿态，放

for y in range(-100, 100, 50):
    Move(140, y, 20, speed)
    Suction_On()
    Move(140, y, 100, speed)
    angle = random.randint(10, 170)
    swift.set_wrist(angle = angle, speed = 1)
    print(angle)
    Move(140, y, 20, speed)
    Suction_Off()

#程序运行计时
from timeit import default_timer as timer
# for ti in range(100, -100, -10):
#     x = 150
#     y = ti
#     Move(x, y, 30, 100)
#     s = time.time()
#     swift.set_position(x + 50, y, 30, 100)
#     e = time.time()
#     print("用时",e - s)

#使用30Pin针脚
# swift.set_digital_direction(pin = 52,value=1) #设置30Pin数字口D52为输出
# swift.set_digital_output(pin = 52, value=1)   #设置数字口D52为高电平
# time.sleep(2)
# swift.set_digital_output(pin = 52, value=0)   #设置数字口D52为低电平

#使用末端limit_switch传感器
# speed = 10
# swift.set_acceleration(2)
# while swift.get_limit_switch() == False and Z >= 10:
#     Move(150, 0, Z, speed)
#     Z = Z - 2
# Suction_On()
# time.sleep(0.2)
# Move(150, 0, Z + 50, speed)
# time.sleep(0.2)
# Suction_Off()
