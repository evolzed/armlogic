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
intrinsicMatrix = np.matrix([[1100.3497, -0.2253, 595.6410, 0.0], [0.0, 1100.3214, 544.8440, 0.0], [0.0, 0.0, 1.0, 0.0]])
extrinsicMatrix = np.matrix([[0.9994, 0.0315, -0.0176, -61.0168], [-0.0295, 0.9943, 0.1027, -93.4316], [0.0207, -0.1021, 0.9946, 523.6375], [0.0, 0.0, 0.0, 1.0]])
transferMatrix = np.dot(intrinsicMatrix, extrinsicMatrix)
coefficientMatrix = transferMatrix[0:3, 0:3]
equaValue = np.matrix([Zc*u-transferMatrix[0,4]],[Zc*v-transferMatrix[1,4]],[Zc-transferMatrix[2,4]])
xWorld = coefficientMatrix.I*equaValue

print(xWorld)



