#!/usr/bin/env python3
# Author: LouYufeng <502550265@qq.com>

import numpy as np

def Px2World(u, v, zc, intrinsicMtx, extrinsicMtx):
    '''
    :param u: 像素坐标
    :param v: 像素坐标
    :param zc: 比例因子
    :param intrinsicMtx: 内参数矩阵
    :param extrinsicMtx: 外参数矩阵
    :return: pwMtx:世界坐标系下的坐标列向量[X,Y,Z]T
    '''
    transferMtx = np.dot(intrinsicMtx, extrinsicMtx)  # 转换矩阵（内参矩阵*外参矩阵）
    coefficientMtx = transferMtx[0:3, 0:3]  # 方程组的系数矩阵
    resultMtx = -transferMtx[0:3, 3] + np.array([[u, v, 1]])*zc  # 方程组移项后，等号右边的矩阵
    resultMtx = resultMtx.reshape(3, 1)  # 转换成列向量
    pwMtx = np.dot(np.linalg.inv(coefficientMtx), resultMtx)  # 世界坐标系下的位置矩阵
    return pwMtx


U = 966  # 待求的像素坐标
V = 777  # 待求的像素坐标
Zc = 570.2425  # 比例因子
# 内参数矩阵
IntrinsicMtx = np.array([[1100.3497, -0.2253, 595.6410, 0.0],
                         [0.0, 1100.3214, 544.8440, 0.0],
                         [0.0, 0.0, 1.0, 0.0]])
# 外参数矩阵
ExtrinsicMtx = np.array([[0.9994, 0.0315, -0.0176, -61.0168],
                         [-0.0295, 0.9943, 0.1027, -93.4316],
                         [0.0207, -0.1021, 0.9946, 523.6375],
                         [0.0, 0.0, 0.0, 1.0]])
PositionMtx = Px2World(U, V, Zc, IntrinsicMtx, ExtrinsicMtx)  # 调用举例
print("PwMtx =\n", PositionMtx)





