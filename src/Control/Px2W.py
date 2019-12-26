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




# 内参数矩阵
IntrinsicMtx = np.array([[1.134364021498704e+03, 0.0, 6.161364934322572e+02, 0.0],
                         [0.0, 1.132974556060971e+03, 5.561652331573952e+02, 0.0],
                         [0.0, 0.0, 1.0, 0.0]])
# 外参数矩阵
ExtrinsicMtx = np.array([[0.998164450924708, 0.057618607890134, -0.018650065280829, -1.526237021849280e+02],
                         [-0.057340832759324, 0.998240439069402, 0.015101480225268, -2.109557849387723e+02],
                         [0.019487375622268, -0.014004350442986, 0.999712018713403, 1.404789115280260e+03],
                         [0.0, 0.0, 0.0, 1.0]])

Zc = 1428.595  # 比例因子
if __name__ == '__main__':
    U1 = 532  # 待求的像素坐标
    V1 = 364  # 待求的像素坐标
    U2 = 476  # 待求的像素坐标
    V2 = 522  # 待求的像素坐标
    PositionMtx1 = Px2World(U1, V1, Zc, IntrinsicMtx, ExtrinsicMtx)  # 调用举例
    PositionMtx2 = Px2World(U2, V2, Zc, IntrinsicMtx, ExtrinsicMtx)  # 调用举例
    Length = np.sqrt((PositionMtx2[0]-PositionMtx1[0])**2+(PositionMtx2[1]-PositionMtx1[1])**2)
    print("Length =\n", Length)




