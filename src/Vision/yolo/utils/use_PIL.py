import os
import random
from datetime import datetime

from PIL import Image, ImageDraw, ImageEnhance
import numpy as np

n = 0  # 记录图片的数量


class Aug(object):
    """利用图片贴合的方式生成数据集"""
    def __init__(self, src, bg, btype, img_path, anno_path):
        # self.init_bg = bg
        self.src = src  # <Image>:需要贴合的bottle
        self.bg = bg  # <Image>:背景
        self.cp_bg = None  # <Image>:拷贝一份背景用来处理，防止对原始背景图片的污染
        # self.cp_bg_gama = None  # <Image>:拷贝一份背景用来处理，防止对原始背景图片的污染
        self.box = None  # tuple:src图片中物体的位置(xmin, ymin, xmax, ymax)
        self.bgbox = None  # tuple:贴合后，bottle位于背景中的位置(xmin, ymin, xmax, ymax)
        self.rotated_img = None  # <Image>:转换后的图片
        self.type = btype  # int/str:标注时的瓶子类别(0-21)
        # self.n = 0  # int:用于防止图片名称重复
        self.file_name = None  # str:文件不带后缀的名称，如_img000.jpg,则file_name=_img000
        self.center = None  # tuple:eg->(200, 400)
        self.img_path = img_path
        self.anno_path = anno_path

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
        # 画完框之后，显示，注意：如果开启显示，执行速度降低而且要保证你的内存足够大。生产使用，建议关闭
        self.cp_bg.show()
        del draw

    def img_txt(self):
        self.bgbox = [self.center[0] + self.box[0],
                      self.center[1] + self.box[1],
                      self.center[0] + self.box[2],
                      self.center[1] + self.box[3]]
        # 处理物体超出边界的情况
        self.bgbox[0] = self.bgbox[0] if 0 <= self.bgbox[0] else 0
        self.bgbox[1] = self.bgbox[1] if 0 <= self.bgbox[1] else 0
        self.bgbox[2] = self.bgbox[2] if self.bgbox[2] <= self.bg.size[0] else self.bg.size[0]
        self.bgbox[3] = self.bgbox[3] if self.bgbox[3] <= self.bg.size[1] else self.bg.size[1]
        with open(self.anno_path + "/" + self.file_name + ".txt", "w") as f:
            f.write(str(int(self.bgbox[0])) + "," + str(int(self.bgbox[1])) + "," +
                    str(int(self.bgbox[2])) + "," + str(int(self.bgbox[3])) + "," + str(self.type))

    def random_center(self):
        # 随机生成贴合中心点
        x_random = random.randint(50, 1100)
        # x_random = random.randint(1150, 1200)
        y_random = random.randint(350, 550)
        # y_random = random.randint(800, 900)
        self.center = (x_random, y_random)

    def stack_up(self):
        """
        图片贴合并进行保存

        :param src_img: 需要贴合的图片
        :param bg_img: 背景图
        :return:
        """
        global n
        # dst.paste(src_img, (100, 400), src_img)
        # self.cp_bg = self.bg.copy()  # 进行gama处理的时候已经赋值self.cp_bg,此处可以关闭
        # 随机生成贴合中心点坐标
        self.random_center()
        self.cp_bg.paste(self.rotated_img, self.center, self.rotated_img)
        # 生成文件名，尽量保证唯一
        t = datetime.strftime(datetime.today(), "%Y%m%d%H%M%S")
        self.cp_bg.save(self.img_path + "/" + t + str(n) + ".jpg")
        print("已保存类别{}，{}张!".format(self.type, n+1))
        self.file_name = t + str(n)
        # 生成txt标注文件
        self.img_txt()
        n += 1

    def img_gama(self):
        """
        对背景和bottle进行随机亮度调整，然后进行贴合


        :return:
        """
        self.cp_bg = self.bg.copy()
        # 对bg进行处理，随机生成0.6-1.5之间的factor，调整bg的亮度
        self.cp_bg = ImageEnhance.Brightness(self.cp_bg).enhance(round(random.uniform(0.6, 1.5), 1))
        # 对瓶子进行处理
        self.rotated_img = ImageEnhance.Brightness(self.rotated_img).enhance(round(random.uniform(0.6, 1.5), 1))


def main(img_num):
    """
    主程序

    :param img_num: 需要生成的图片数量
    :return:
    """
    aug = Aug(src, bg, btype, img_save_path, anno_save_path)
    for _ in range(img_num):
        # if not angle % 2 == 0:
        #     continue
        # 旋转图片
        aug.rotate_src(random.randint(0, 360))
        # 获得旋转后物体的位置
        aug.get_obj_box()
        # 对旋转后的图片进行亮度调整
        aug.img_gama()
        # 进行贴合，并保存物体在背景下的坐标
        aug.stack_up()
        # 进行画框，测试时用。注意：如果开启显示，执行速度降低而且要保证你的内存足够大。生产使用，建议关闭
        # aug.draw_box()


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
    img_save_path = "dataset_13"
    anno_save_path = "anno_13"
    img_total_num = 1000
    # 生成的图片统一保存到dataset文件夹
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)
    if not os.path.exists(anno_save_path):
        os.mkdir(anno_save_path)
    # 将需要的物体存放列表
    bottle_list = ["01", "02", "03", "04", "05"]
    bottle_num = len(bottle_list)
    for i in range(bottle_num):
        # bottle need to paste
        src_path = "images/bottle_01_{}.png".format(bottle_list[i])
        # background
        bg_path = "images/bg.jpg"
        btype = int(src_path.split("_")[1])
        src = Image.open(src_path).convert("RGBA")
        # background
        bg = Image.open(bg_path)
        # 根据总数量，计算每个图片需要的个数。保证数据均衡
        img_num = img_total_num // bottle_num
        # 如果不能整除，这一步处理，统一添加到最后一张图上
        if i == bottle_num - 1:
            img_num = img_num + img_total_num % bottle_num
        main(img_num)
