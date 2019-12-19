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
import functools
import threading
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from uarm.wrapper import SwiftAPI

swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'}, enable_handle_thread=False)
# swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'}, enable_write_thread=True)
# swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'}, enable_report_thread=True)
# swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'}, enable_handle_thread=True, enable_write_thread=True, enable_report_thread=True)
swift.waiting_ready() #等待手臂完成初始化

print(swift.get_device_info()) #机械臂固件4.3.2版本为高性能模式，如需要使用激光雕刻/3D打印等功能，请将固件降为3.2.0版本，升级/降级教程参考：https://pan.baidu.com/s/16F-tPocukHLSP4yHc-L1Mw 提取码 394p

swift.send_cmd_async('M2400 S0')  # 设置机械臂工作模式，为S0常规模式
swift.send_cmd_async('M2123 V1')  # 开启失步检测功能

resetSpeed = 500
speed = 500   #弧线运动的速度，500，底座左右极限750
swift.set_acceleration(10)  #设置加速度，20，底座左右极限35

n = 0
DeltaY = 40
Y = 100
def Reset():
    swift.set_position(150,0,100,speed = resetSpeed,wait=False,timeout=10,cmd='G0')#G0模式为两点之间以最快方式到达的模式，G1模式为两点之间以直线到达的模式
    swift.flush_cmd()#清除缓存，保证前序指令一定执行完后，再执行下一个指令。
    time.sleep(1)

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