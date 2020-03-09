# Standard imports
import cv2
import numpy as np

# Read images
# src = cv2.imread("images/bottle_01.png")
src = cv2.imread("images/12.jpg")
dst = cv2.imread("images/bg.jpg")

# Create a rough mask around the airplane.
src_mask = np.zeros(src.shape, src.dtype)

# 也可以不需要下面两行，只是效果差一点。
# 不使用的话我们得将上面一行改为 mask = 255 * np.ones(obj.shape, obj.dtype) <-- 全白
poly = np.array([[4, 80], [30, 54], [151, 63], [254, 37], [298, 90], [272, 134], [43, 122]], np.int32)
cv2.fillPoly(src_mask, [poly], (255, 255, 255))

# 这是 瓶子 CENTER 所在的地方
center = (800, 100)

# Clone seamlessly.
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)

# 保存结果
cv2.imwrite("images/result1.jpg", output)
