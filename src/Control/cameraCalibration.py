#!/usr/bin/env python3
# Author: LouYufeng <502550265@qq.com>

import os
import sys
import time
import math
import random
import numpy as np
# sys.path.append('../../')
Zc = 570.2425
u = 966.0
v = 777.0
intrinsicMtx = np.array([[1100.3497, -0.2253, 595.6410, 0.0], [0.0, 1100.3214, 544.8440, 0.0], [0.0, 0.0, 1.0, 0.0]])
extrinsicMtx = np.array([[0.9994, 0.0315, -0.0176, -61.0168], [-0.0295, 0.9943, 0.1027, -93.4316], [0.0207, -0.1021, 0.9946, 523.6375], [0.0, 0.0, 0.0, 1.0]])
transferMtx = np.dot(intrinsicMtx, extrinsicMtx)
coefficientMtx = transferMtx[0:3, 0:3]
resultMtx = -transferMtx[0:3, 3] + np.array([[u, v, 1]])*Zc #行向量
resultMtx = resultMtx.reshape(3, 1) #转换成列向量
Xw = np.dot(np.linalg.inv(coefficientMtx), resultMtx)
print("Xw =\n",Xw)







