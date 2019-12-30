#!/usr/bin/env python3
# Author: LouYufeng <502550265@qq.com>


import sys
import os
import math
import numpy as np
import operator

sys.path.insert(0, os.path.abspath("../../"))

# from src.Control.Px2W import *
from src.Control.Px2W import Zc, IntrinsicMtx, ExtrinsicMtx, Px2World
from lib.uArmSDK.uarm.wrapper import SwiftAPI

Robot = 1
# Robot = 0

#机械臂初始化
if Robot == 1:
    swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'}, enable_handle_thread=False)
    swift.waiting_ready() #等待手臂完成初始化
    swift.send_cmd_async('M2400 S0')  # 设置机械臂工作模式，为S0常规模式
    swift.send_cmd_async('M2123 V1')  # 开启失步检测功能
    swift.set_digital_direction(pin = 32,value=1) #设置30Pin数字口D25为输出
    speed = 300  # 弧线运动的速度，500
    resetSpeed = 50
    swift.set_acceleration(15)  # 设置加速度，20


def Rz(theta):
    '''
    :param theta: 输入绕z轴转角，右手螺旋定则
    :return: 旋转矩阵
    '''
    RotationMtx = np.array([[math.cos(theta), -math.sin(theta), 0],
              [math.sin(theta), math.cos(theta), 0],
              [0, 0, 1]])
    return RotationMtx


# BottleType, u, v, angle2Xw, BottleHeight,Xw, Yw, Zw, distance2uArm |模拟矩阵，angle2Xw的取值范围为-90°到90°
#     0       1  2      3          4       5    6   7        8
bottleDict = {"box": [["Bottle1", 830, 450, 30, 100],
                      ["Bottle3", 810, 450, 20, 110],
                      ["Bottle2", 790, 450, -50, 95],
                      ["Bottle1", 770, 450, 60, 100],
                      ["Bottle2", 750, 450, -10, 98],
                      ["Bottle1", 730, 450, -15, 98],
                      ["Bottle3", 710, 450, -25, 100],
                      ["Bottle1", 690, 450, 35, 100]], "ABC1": [1, 2, 3], "ABC2": [1, 2, 3]}
PI = 3.1415926
uArmbottom_P = [0, 0, 0]  # 机械臂基坐标系下，机械臂底座中心的位置，单位mm
world2uarm_P = np.array([[350], [-325], [0]])  # 世界坐标系原点相在机械臂基坐标系的位置矢量，单位mm
world2uarm_R = PI/2  # 世界坐标系绕z轴转90度(实际可能不是准确的90°)后，与机械臂基坐标系重合，用来生成旋转矩阵

Px = bottleDict["box"]  # 瓶子信息矩阵

numRow = np.shape(Px)  # 瓶子数目
ratio = 0.5  # 宽度代替高度的比例因子
#  从像素坐标计算到世界坐标系下的坐标，计算位置矢量，计算物理距离，并放入瓶子信息矩阵
for i in range(0, numRow[0]):
    world_P = Px2World(Px[i][1], Px[i][2], Zc, IntrinsicMtx, ExtrinsicMtx)  # 根据像素信息计算瓶子在世界坐标系下的位置矢量
    uarm_P = world2uarm_P + np.dot(Rz(world2uarm_R), world_P)  # 机械臂基坐标系下的坐标
    distance = float(np.sqrt((uarm_P[0] - uArmbottom_P[0]) ** 2 + (uarm_P[1] - uArmbottom_P[1]) ** 2))  # 计算瓶子到机械臂底座中心之间的距离
    for j in range(0, 2):
        Px[i].append(float(uarm_P[j]))  # 将位置矢量(Z分量不使用)添加到矩阵
    Px[i].append(Px[i][4]*ratio)  # 按比例计算瓶子高度
    Px[i].append(distance)  # 将距离添加到矩阵


Px.sort(key=operator.itemgetter(8), reverse=False)  # 对矩阵的第八列距离值进行升序排列，距离最近的先取
# print(Px)


if Robot == 1:
    # 吸，调姿态，放,电磁继电器的CH3可以正常使用，其他可能会有问题
    swift.set_position(150, 0, 100, speed=resetSpeed, wait=True, timeout=10,cmd='G0')  # 初始化机器人末端执行器位置
    swift.flush_cmd()  # 清除缓存，保证前序指令一定执行完后，再执行下一个指令。

    for i in range(0, numRow[0]):  # 依次从距离远近取瓶子操作
        swift.set_wrist(angle=90)  # 初始化舵机位置
        swift.set_position(Px[i][5], Px[i][6], Px[i][7], speed=speed, wait=True, cmd='G0')  # 到达目标位置
        # swift.set_digital_output(pin=32, value=1, wait=True, timeout=10)  # 吸盘吸
        swift.set_position(Px[i][5], Px[i][6], Px[i][7] + 50, speed=speed, wait=True, cmd='G0')  # 把瓶子举高50mm
        bottomAngle0 = swift.get_servo_angle(servo_id=0)  # 获取底座关节角0,以机械臂Y轴正方向为基准
        swift.set_position(150, Px[i][6], Px[i][7] + 50, speed=speed, wait=True, cmd='G0')  # 把瓶子移动到靠近边缘的轨道上空
        bottomAngle1 = swift.get_servo_angle(servo_id=0)  # 获取底座关节角1
        swift.set_wrist(angle=90 + (Px[i][3] + (bottomAngle1 - bottomAngle0)))  # 根据姿态角计算舵机转角并发送给舵机执行
        swift.set_position(150, Px[i][6], Px[i][7], speed=speed, wait=True, cmd='G0')  # 把瓶子移动到靠近边缘的轨道
        # swift.set_digital_output(pin=32, value=0, wait=True, timeout=10)  # 吸盘放下瓶子






