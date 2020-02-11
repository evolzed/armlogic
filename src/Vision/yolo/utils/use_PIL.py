import os
from datetime import datetime

from PIL import Image, ImageDraw
import numpy as np


class Aug(object):
    """利用图片贴合的方式生成数据集"""
    def __init__(self, src, bg, btype):
        # self.init_bg = bg
        self.src = src
        self.bg = bg
        self.cp_bg = None
        self.box = None
        self.bgbox = None
        self.rotated_img = None
        self.type = btype
        self.n = 0
        self.file_name = None
        self.center = (100, 400)

    def rotate_src(self, angle):
        """
        对图像进行旋转处理

        :param angle: 需要旋转的角度
        :return: 旋转后的图像
        """
        # 旋转angle后
        self.rotated_img = self.src.rotate(angle, expand=True)
        # src_img.save("images/rotated_120.png")
        return self.rotated_img

    def get_obj_box(self):
        """
        获得旋转后的图片位置框信息

        :param src_img: 需要获得物体位置的图片，注意，一定要是处理后的透明背景rgba格式的图片
        :return: box：返回物体的坐标信息
        """
        # w, h = src_img.size
        # print(src_img.size)  # (w, h)
        # 图片数组化，注意array和asarray的区别
        arr = np.array(self.rotated_img)
        # print(arr.shape)  # (206, 106, 4)  : h, w, c
        # 计算图像的边界框
        arr[:, :, :-1] = 0
        # print(arr)
        # 获取非零的元素的索引
        b = np.nonzero(arr)
        # print(b)
        # 将索引转换成numpy数组
        b_arr = np.asarray(b)
        # print(b_arr)
        # 获取第一个维度中第一个和最后一个出现非0的坐标
        b_arr_min_0 = b_arr[0][0]
        b_arr_max_0 = b_arr[0][-1]
        # print(b_arr_min_0)
        # print(b_arr_max_0)
        # 统计第一个维度种第一个和最后一个非零个数，然后找出最大和最小值，作为box的left-up，和right-bottom
        # b_arr_min_count = np.sum(b_arr[0] == b_arr_min_0)
        # b_arr_max_count = np.sum(b_arr[0] == b_arr_max_0)
        # print(b_arr_min_count)
        # print(b_arr_max_count)
        # 截取对应的坐标中的对应元素
        # print(b_arr[1])
        b_arr_min_1 = np.min(b_arr[1])
        b_arr_max_1 = np.max(b_arr[1])
        # print(b_arr_min_1)
        # print(b_arr_max_1)
        # 输出left-up，right-bottom坐标
        self.box = (b_arr_min_1, b_arr_min_0, b_arr_max_1, b_arr_max_0)
        # print("box", box)
        # print("box", (b_arr_min_0, b_arr_min_1, b_arr_max_0, b_arr_max_1))
        # return self.box

    def draw_box(self):
        # 根据box坐标进行画框
        draw = ImageDraw.Draw(self.cp_bg)
        # draw.line()
        draw.rectangle(self.bgbox, outline="red", width=2)
        # 画完框之后，显示
        self.cp_bg.show()
        # self.bg = self.init_bg
        # self.bg.show()
        del draw

    def img_txt(self):
        self.bgbox = (self.center[0] + self.box[0],
                      self.center[1] + self.box[1],
                      self.center[0] + self.box[2],
                      self.center[1] + self.box[3])
        with open("anno/" + self.file_name + ".txt", "w") as f:
            f.write(str(int(self.bgbox[0])) + "," + str(int(self.bgbox[1])) + "," +
                    str(int(self.bgbox[2])) + "," + str(int(self.bgbox[3])) + "," + str(self.type))

    def stack_up(self):
        """
        图片贴合并进行保存

        :param src_img: 需要贴合的图片
        :param bg_img: 背景图
        :return:
        """
        # dst.paste(src_img, (100, 400), src_img)
        self.cp_bg = self.bg.copy()
        self.cp_bg.paste(self.rotated_img, self.center, self.rotated_img)
        # 生成文件名，尽量保证唯一
        t = datetime.strftime(datetime.today(), "%Y%m%d%H%M%S")
        self.cp_bg.save("dataset/" + t + str(self.n) + ".jpg")
        self.file_name = t + str(self.n)
        # todo 生成txt文件
        self.img_txt()
        self.n += 1


def main():
    aug = Aug(src, bg, btype)
    for i in range(5):
        angle = (i + 1) * 10
        aug.rotate_src(angle)
        aug.get_obj_box()
        aug.stack_up()
        aug.draw_box()


def get_xy():
    """
    用于测试rgba模式

    :return: None
    """
    dst = Image.open("images/bg.jpg").convert("RGBA")
    # 旋转目标图片
    dst = dst.rotate(45, expand=True)
    w1, h2 = dst.size
    color_0 = (0, 0, 0)
    for m in range(w1):
        for n in range(h2):
            dot = (m, n)
            color_1 = dst.getpixel(dot)
            if color_1 == color_0:
                color_1 = color_1[:-1] + (0,)
                dst.putpixel(dot, color_1)
    # 遍历结束，已经修改dst完成，返回
    return dst


if __name__ == '__main__':
    btype = 1
    # 生成的图片统一保存到dataset文件夹
    if not os.path.exists("dataset"):
        os.mkdir("dataset")
    if not os.path.exists("anno"):
        os.mkdir("anno")
    # bottle need to paste
    src = Image.open("images/bottle_01.png").convert("RGBA")
    # background
    bg = Image.open("images/bg.jpg")
    main()

