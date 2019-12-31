#!/usr/bin/env python3
# Author: Lou Yufeng <502550265@qq.com>


import sys
import os
import math
import numpy as np
import operator

sys.path.insert(0, os.path.abspath("../../"))

# from src.Control.Px2W import *
from src.Control.Px2W import Zc, IntrinsicMtx, ExtrinsicMtx, Px2World
from lib.uArmSDK.uarm.wrapper import SwiftAPI

def Rz(theta):
    '''
    绕z轴旋转的旋转矩阵函数，表达式形式固定，输入转角自动生成矩阵，
    左乘一个矢量，可以将一个矢量转换到新坐标系下。

    :param theta: 输入绕z轴转角，右手螺旋定则
    :return: 旋转矩阵
    '''
    RotationMtx = np.array([[math.cos(theta), -math.sin(theta), 0],
                            [math.sin(theta), math.cos(theta), 0],
                            [0, 0, 1]])
    return RotationMtx

def ResetRobotPosition():
    '''
    机械臂初始化位置函数，运动最大速度50，使末端运动到（150，0，100）。

    :return:  None
    '''
    swift.set_position(150, 0, 100, speed=50, wait=True, timeout=10, cmd='G0')  # 初始化机器人末端执行器位置
    swift.flush_cmd()  # 清除缓存，保证前序指令一定执行完后，再执行下一个指令。



# RobotOn = 1  # 机械臂已连接
RobotOn = 0  # 机械臂未连接


# BottleType, u, v, angle2Xw, BottleHeight,Xw, Yw, Zw, isInside |模拟矩阵，angle2Xw的取值范围为-90°到90°
#     0       1  2      3          4       5    6   7      8
bottleDict = {"box": [["Bottle1", 745, 439, -45, 55],
                      ["Bottle2", 896, 437, -45, 55],
                      ["Bottle3", 631, 440, 45, 55]], "ABC1": [1, 2, 3], "ABC2": [1, 2, 3]}
PI = 3.1415926  # 圆周率
uArmbottom_P = [0, 0, 0]  # 机械臂基坐标系下，机械臂底座中心的位置，单位mm

world2uarm_R = 0.47*PI  # 世界坐标系绕z轴转90度(实际可能不是准确的90°)后，与机械臂基坐标系重合，用来生成旋转矩阵
# world2uarm_P = np.array([[290], [-335], [0]])  # 世界坐标系原点相在机械臂基坐标系的位置矢量，单位mm,uv=750，630
# uarmMarker_Puv = np.array([750, 630])
world2uarm_P = -Px2World(750, 630, Zc, IntrinsicMtx, ExtrinsicMtx)  # 机械臂基坐标系原点像素坐标系中的坐标转换成世界坐标系下的平移矢量，加负号转换成机械臂基坐标系下的矢量
world2uarm_PR = np.dot(Rz(world2uarm_R), world2uarm_P)  # 旋转坐标轴

bottleInfo = bottleDict["box"]  # 瓶子信息矩阵
ratio = 1.2  # 宽度代替高度时，像素映射到实际物理距离的比值
numRow = np.shape(bottleInfo)  # 瓶子数目


#  从像素坐标计算到世界坐标系下的坐标，计算位置矢量，计算物理距离，并放入瓶子信息矩阵
for i in range(0, numRow[0]):
    world_P = Px2World(bottleInfo[i][1], bottleInfo[i][2], Zc, IntrinsicMtx, ExtrinsicMtx)  # 根据像素信息计算瓶子在世界坐标系下的位置矢量
    uarm_P = world2uarm_PR + np.dot(Rz(world2uarm_R), world_P)  # 机械臂基坐标系下的坐标
    distance2uArm = float(np.sqrt((uarm_P[0] - uArmbottom_P[0]) ** 2 + (uarm_P[1] - uArmbottom_P[1]) ** 2))  # 计算瓶子到机械臂底座中心之间的距离
    for j in range(0, 2):
        bottleInfo[i].append(int(uarm_P[j]))  # 将位置矢量(Z分量不使用)添加到矩阵，转化成整形，否则数据太长，机械臂接收存不下
    bottleHeight = int(bottleInfo[i][4] * ratio)  # 按比例计算瓶子高度
    bottleInfo[i].append(bottleHeight)  # 将瓶子高度添加进数组
    # 机械臂工作空间筛选
    if distance2uArm <= 100 or distance2uArm >= 340:  # 去除圆环外的区域
        isInside = 0  # 0/1代表是否在规定的工作区间内
    elif uarm_P[0] <= 100:  # 去除X<=100的区域
        isInside = 0
    elif bottleHeight <= 30 or bottleHeight >= 200:  # 去除Z太低和太高的区域
        isInside = 0
    else:
        isInside = 1
    bottleInfo[i].append(isInside)


bottleInfo.sort(key=operator.itemgetter(6), reverse=True)  # 对矩阵的第六列的值（Y）进行升序排列，数值大的先取
# print(bottleInfo)


if RobotOn == 1:
    swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'}, enable_handle_thread=False)
    swift.waiting_ready()  # 等待手臂完成初始化
    swift.send_cmd_async('M2400 S0')  # 设置机械臂工作模式，为S0常规模式
    swift.send_cmd_async('M2123 V1')  # 开启失步检测功能
    swift.set_digital_direction(pin=32, value=1)  # 设置30Pin数字口D32为输出(1)
    speed = 250  # 弧线运动的速度，500
    swift.set_acceleration(10)  # 设置加速度，20

    ResetRobotPosition()  # 机械臂初始化末端位置
    # 吸，调姿态，放,电磁继电器的CH3可以正常使用，其他可能会有问题
    for i in range(0, numRow[0]):  # 依次从距离远近取瓶子操作
        if bottleInfo[i][8] == 1:  # 落在可操作区域内，才会执行
            swift.set_wrist(angle=90, wait=True)  # 初始化舵机位置
            swift.set_position(bottleInfo[i][5], -bottleInfo[i][6], bottleInfo[i][7], speed=speed, wait=True)  # 到达目标位置，由于机械臂Y轴与自定义的机械臂基坐标系的Y轴反向，Y的值添加了负号
            swift.set_digital_output(pin=32, value=1, wait=True, timeout=10)  # 吸盘吸
            swift.set_position(bottleInfo[i][5], -bottleInfo[i][6], bottleInfo[i][7] + 25, speed=speed, wait=True)  # 把瓶子举高25mm
            bottomAngle0 = swift.get_servo_angle(servo_id=0, timeout=10)  # 获取底座关节角0,以机械臂Y轴负方向为基准，且取值范围为-90°~90°
            swift.set_position(180, -bottleInfo[i][6], bottleInfo[i][7] + 25, speed=speed, wait=True)  # 把瓶子移动到靠近边缘的轨道上空
            bottomAngle1 = swift.get_servo_angle(servo_id=0, timeout=10)  # 获取底座关节角1
            swift.set_wrist(angle=90 + (bottleInfo[i][3] + (bottomAngle1 - bottomAngle0)), wait=True)  # 根据姿态角计算舵机转角并发送给舵机执行
            swift.set_position(180, -bottleInfo[i][6], bottleInfo[i][7], speed=speed, wait=True)  # 把瓶子移动到靠近边缘的轨道
            swift.set_digital_output(pin=32, value=0, wait=True, timeout=10)  # 吸盘放下瓶子
            swift.set_position(180, -bottleInfo[i][6], bottleInfo[i][7] + 25, speed=speed, wait=True)  # 放完后，上移一些，防止擦到瓶子


