import numpy as np
from PIL import Image, ImageDraw
import cv2


# 构造6个框的坐标
#         x1, y1, x2, y2
points = [
          [2, 5, 105, 120],
          # [68, 79, 245, 350],
          # [55, 120, 360, 480],
          # [120, 160, 400, 450],
          # [33, 200, 160, 300],
          # [180, 280, 260, 399],
          # [400, 600, 800, 900],
            [200, 200, 400, 400],
            [300, 300, 500, 500],
            [600, 400, 900, 600],
            [800, 500, 850, 550],

          ]
coordinates = np.asarray(points)
print(coordinates[0])
print(len(coordinates))
# 利用PIL的new实现
img = Image.new("RGB", (1280, 960), (128, 128, 128))
draw = ImageDraw.Draw(img)
draw.rectangle(list(coordinates[0]), outline=(255, 0, 0), width=4)
draw.rectangle(list(coordinates[1]), outline=(0, 255, 0), width=4)
draw.rectangle(list(coordinates[2]), outline=(0, 0, 255), width=4)
draw.rectangle(list(coordinates[3]), outline=(255, 100, 100), width=4)
draw.rectangle(list(coordinates[4]), outline=(100, 255, 100), width=4)
# draw.rectangle(list(coordinates[5]), outline=(100, 100, 255), width=4)
# 每个box的left-up 和 right-bottom
for i in range(len(coordinates)-1):
    # print(coordinates[i])
    b1_x1 = coordinates[i][0]
    b1_y1 = coordinates[i][1]
    b1_x2 = coordinates[i][2]
    b1_y2 = coordinates[i][3]
    for j in range(len(coordinates[i+1:])):
        b2_x1 = coordinates[i+1:][j][0]
        b2_y1 = coordinates[i+1:][j][1]
        b2_x2 = coordinates[i+1:][j][2]
        b2_y2 = coordinates[i+1:][j][3]
        # 排除大框包裹小框的情况
        if (b1_x1 <= b2_x1 and b1_x2 >= b2_x2 and b1_y1 <= b2_y1 and b1_y2 >= b2_y2) or \
           (b1_x1 >= b2_x1 and b1_x2 <= b2_x2 and b1_y1 >= b2_y1 and b1_y2 <= b2_y2):
            continue
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
        if inter_area:
            draw.rectangle((inter_rect_x1, inter_rect_y1, inter_rect_x2, inter_rect_y2), outline=(255, 255, 255), width=4)
        # calculate the union area
        union_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area
        iou = inter_area / union_area
        print("IOU is %0.2f" % iou)
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