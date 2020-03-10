import random

import numpy as np
from PIL import Image, ImageDraw
import cv2


# 构造6个框的坐标
#         x1, y1, x2, y2
points = [
          [2, 5, 105, 120],
          [68, 79, 245, 350],
          [55, 120, 360, 480],
          [120, 160, 400, 450],
          [33, 200, 160, 300],
          [180, 280, 260, 399],
          [400, 600, 800, 900],
          [200, 200, 400, 400],
          [300, 300, 500, 500],
          [600, 400, 900, 600],
          [800, 500, 850, 550],
          ]
coordinates = np.array(points)
print(coordinates[0])
print(len(coordinates))
# 利用PIL的new实现
img = Image.new("RGB", (1280, 960), (128, 128, 128))
draw = ImageDraw.Draw(img)
# for n in range(len(points)):
#     draw.rectangle(points[n], outline=(random.randrange(256), random.randrange(256), random.randrange(256)), width=4)
# draw.rectangle(list(coordinates[0]), outline=(255, 0, 0), width=4)
# draw.rectangle(list(coordinates[1]), outline=(0, 255, 0), width=4)
# draw.rectangle(list(coordinates[2]), outline=(0, 0, 255), width=4)
# draw.rectangle(list(coordinates[3]), outline=(255, 100, 100), width=4)
# draw.rectangle(list(coordinates[4]), outline=(100, 255, 100), width=4)
# draw.rectangle(list(coordinates[5]), outline=(100, 100, 255), width=4)
# 记录需要删除的box
# rm_boxes = set()
# 每个box的left-up 和 right-bottom
rm_list = []
for i in range(len(coordinates)-1):
    if list(coordinates[i]) in rm_list:
        continue
    # print(coordinates[i])
    b1_x1 = coordinates[i][0]
    b1_y1 = coordinates[i][1]
    b1_x2 = coordinates[i][2]
    b1_y2 = coordinates[i][3]
    cp_coord = coordinates[i+1:]
    for j in range(len(cp_coord)):
        if list(cp_coord[j]) in rm_list:
            continue
        b2_x1 = cp_coord[j][0]
        b2_y1 = cp_coord[j][1]
        b2_x2 = cp_coord[j][2]
        b2_y2 = cp_coord[j][3]
        # 排除大框包裹小框的情况，删除小框
        if b1_x1 <= b2_x1 and b1_x2 >= b2_x2 and b1_y1 <= b2_y1 and b1_y2 >= b2_y2:
            points.remove(list(cp_coord[j]))
            rm_list.append(list(cp_coord[j]))
            continue
        if b1_x1 >= b2_x1 and b1_x2 <= b2_x2 and b1_y1 >= b2_y1 and b1_y2 <= b2_y2:
            # rm_box = random.sample([coordinates[i], coordinates[i+1:][j]], 1)
            # rm_boxes.add(coordinates[i+1:][j])
            # try:
            points.remove(list(coordinates[i]))
            rm_list.append(list(coordinates[i]))
            # except:
            #     pass
            break
        inter_rect_x1 = max(b1_x1, b2_x1)
        inter_rect_y1 = max(b1_y1, b2_y1)
        inter_rect_x2 = min(b1_x2, b2_x2)
        inter_rect_y2 = min(b1_y2, b2_y2)

        # calculate the inter area,如果inter_rect的left-up > right-bottom,则没有交集
        # if (inter_rect_x1 >= inter_rect_x2) or (inter_rect_y1 >= inter_rect_y2):
        #     inter_area = 0
        #     continue
        inter_area = (inter_rect_x2 - inter_rect_x1) * (inter_rect_y2 - inter_rect_y1)
        if (inter_rect_x1 >= inter_rect_x2) or (inter_rect_y1 >= inter_rect_y2):
            inter_area = 0
        # 画出交集框
        # if inter_area:
        #     draw.rectangle((inter_rect_x1, inter_rect_y1, inter_rect_x2, inter_rect_y2),
        #                    outline=(255, 255, 255), width=4)
        # calculate the union area
        union_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area
        iou = inter_area / union_area
        # 设置条件，如果iou超过阈值，只保留一个
        if iou >= 0.2:
            # rm_boxes.add(coordinates[i+1:][j])
            print(list(cp_coord[j]))
            points.remove(list(cp_coord[j]))
            rm_list.append(list(cp_coord[j]))
        print("IOU is %0.2f" % iou)

print("points", points)
for m in range(len(points)):
    draw.rectangle(points[m], outline=(random.randrange(256), random.randrange(256), random.randrange(256)), width=4)
# # cv2构造背景
# bg_arr = np.full(shape=(960, 1280, 3), fill_value=128, dtype="uint8")
# # Image构造背景
# print(bg_arr)
# print(type(bg_arr))
# # cv2.imshow("aa", bg_arr)
# # cv2.waitKey()
# img = Image.fromarray(bg_arr)
# img.show()

# # 利用PIL的new实现
# img = Image.new("RGB", (1280, 960), (128, 128, 128))
# draw = ImageDraw.Draw(img)
# draw.rectangle(list(coordinates[0]), outline=(255, 0, 0), width=2)
# draw.rectangle(list(coordinates[1]), outline=(0, 255, 0), width=2)
# draw.rectangle(list(coordinates[2]), outline=(0, 0, 255), width=2)
# draw.rectangle(list(coordinates[3]), outline=(255, 100, 100), width=2)
img.show()