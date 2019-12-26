#!/usr/bin/env python3
# Author: LouYufeng <502550265@qq.com>


import sys
import os
import numpy as np
# print(sys.path)
sys.path.insert(0, os.path.abspath("../../"))
# print(sys.path)

# from src.Control.Px2W import *
from src.Control.Px2W import Zc, IntrinsicMtx, ExtrinsicMtx, Px2World

bottleDict = {"box": [["Bottle1", 100, 200],
                      ["Bottle3", 110, 250],
                      ["Bottle2", 150, 250],
                      ["Bottle1", 160, 250],
                      ["Bottle2", 150, 250],
                      ["Bottle1", 150, 250],
                      ["Bottle3", 150, 250],
                      ["Bottle1", 150, 250]],
                        "ABC1": [1, 2, 3],
                        "ABC2": [1, 2, 3]}
Px = bottleDict["box"]
# print("Px = ", Px)

PositionMtx1 = Px2World(Px[2][1], Px[2][2], Zc, IntrinsicMtx, ExtrinsicMtx)  # 调用举例
PositionMtx2 = Px2World(Px[3][1], Px[3][2], Zc, IntrinsicMtx, ExtrinsicMtx)  # 调用举例
Length = np.sqrt((PositionMtx2[0]-PositionMtx1[0])**2+(PositionMtx2[1]-PositionMtx1[1])**2)
print("Length =\n", Length)
