import os
import random
from datetime import datetime

from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import numpy as np
import imgaug.augmenters as iaa

n = 0  # 记录图片的数量


class Aug(object):

    class_bg = None
    """利用图片贴合的方式生成数据集"""
    def __init__(self, src, bg, btype, img_path, anno_path):
        # self.init_bg = bg
        self.src = src  # <Image>:需要贴合的bottle
        self.bg = bg  # <Image>:背景
        self.bg_prod = None  # 增强过后的背景
        self.bg_arr = np.array(self.bg)  # 将背景图转化为[cv2格式, 未涉及通道变换，可选转通道]的ndarray
        # self.bg_prod = None  # <Image>:拷贝一份背景用来处理，防止对原始背景图片的污染
        # self.bg_prod_gama = None  # <Image>:拷贝一份背景用来处理，防止对原始背景图片的污染
        self.box = None  # tuple:src图片中物体的位置(xmin, ymin, xmax, ymax)
        self.box_in_bg = None  # tuple:贴合后，bottle位于背景中的位置(xmin, ymin, xmax, ymax)
        self.rotated_img = None  # <Image>:转换后的图片
        self.type = btype  # int/str:标注时的瓶子类别(0-21)
        # self.n = 0  # int:用于防止图片名称重复
        self.file_name = None  # str:文件不带后缀的名称，如_img000.jpg,则file_name=_img000
        self.center = None  # tuple:eg->(200, 400)
        self.img_path = img_path
        self.anno_path = anno_path
        self.seq = iaa.OneOf([
            iaa.Crop(px=(0, 150)),  # 随机从图像的每个边裁剪掉0~100个像素值，保持图像原尺寸不变
            iaa.Fliplr(0.4),  # 有0.5的概率对图像进行左右翻转
            iaa.Flipud(0.4),  # 有0.5的概率对图像进行上下翻转
            iaa.GaussianBlur(sigma=(0, 3.0)),  # 进行高斯模糊，范围0【不变】~3.0【最大】
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # x，y缩放比例
                # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # x,y方向上的偏移范围是-0.2~0.2
                # rotate=(0, 360),  # 随机旋转范围
            ),
            iaa.Affine(
                # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # x，y缩放比例
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # x,y方向上的偏移范围是-0.2~0.2
                # rotate=(0, 360),  # 随机旋转范围
            ),
            # 指定两次旋转，增大随机概率
            iaa.Affine(
                # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # x，y缩放比例
                # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # x,y方向上的偏移范围是-0.2~0.2
                rotate=(0, 360),  # 随机旋转范围
                mode="edge",
            ),
            iaa.Affine(
                # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # x，y缩放比例
                # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # x,y方向上的偏移范围是-0.2~0.2
                rotate=(0, 360),  # 随机旋转范围
                mode="edge",
            ),
        ])

    def img_augm(self):
        """对背景进行随机增强处理"""
        _bg_arr = self.seq.augment_image(self.bg_arr)
        # 增强后的bg_prod用于背景贴合
        self.bg_prod = Image.fromarray(_bg_arr)
        # 定义类属性
        Aug.class_bg = self.bg_prod

    def rotate_src(self):
        """
        对图像进行旋转处理

        :param angle: 需要旋转的角度
        :return: 旋转后的图像
        """
        # if isinstance(self.src, list):
        #     self.rotated_img = [i.rotate(random.randint(0, 360), expand=True) for i in self.src]
        #     return self.rotated_img
        # 旋转angle后
        self.rotated_img = self.src.rotate(random.randint(0, 360), expand=True)
        # src_img.save("images/rotated_120.png")
        return self.rotated_img

    def get_obj_box(self):
        """
        获得旋转后的图片位置框信息

        :param src_img: 需要获得物体位置的图片，注意，一定要是处理后的透明背景rgba格式的图片
        :return: box：返回物体的坐标信息
        """
        # if isinstance(self.rotated_img, list):
        #     self.box = [self.handle_box(i) for i in self.rotated_img]
        #     return self.box
        self.box = self.handle_box(self.rotated_img)
        return self.box

    @staticmethod
    def handle_box(rotated_img):
        """获得旋转后的图片标注框"""
        # w, h = src_img.size
        # print(src_img.size)  # (w, h)
        # 图片数组化，注意array和asarray的区别
        arr = np.array(rotated_img)
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
        box = (b_arr_min_1, b_arr_min_0, b_arr_max_1, b_arr_max_0)
        return box
        # print("box", box)
        # print("box", (b_arr_min_0, b_arr_min_1, b_arr_max_0, b_arr_max_1))
        # return self.box

    def draw_box(self):
        # 根据box坐标进行画框
        draw = ImageDraw.Draw(self.bg_prod)
        # draw.line()
        draw.rectangle(self.box_in_bg, outline="red", width=2)
        # 画完框之后，显示，注意：如果开启显示，执行速度降低而且要保证你的内存足够大。生产使用，建议关闭
        self.bg_prod.show()
        del draw

    def trans_coord(self):
        """将中心坐标和bbox坐标转化成全局坐标"""
        self.box_in_bg = [self.center[0] + self.box[0],
                          self.center[1] + self.box[1],
                          self.center[0] + self.box[2],
                          self.center[1] + self.box[3]]
        return self.box_in_bg

    def img_txt(self):
        # 处理物体超出边界的情况
        self.box_in_bg[0] = self.box_in_bg[0] if 0 <= self.box_in_bg[0] else 0
        self.box_in_bg[1] = self.box_in_bg[1] if 0 <= self.box_in_bg[1] else 0
        self.box_in_bg[2] = self.box_in_bg[2] if self.box_in_bg[2] <= self.bg.size[0] else self.bg.size[0]
        self.box_in_bg[3] = self.box_in_bg[3] if self.box_in_bg[3] <= self.bg.size[1] else self.bg.size[1]
        with open(self.anno_path + "/" + self.file_name + ".txt", "w") as f:
            f.write(str(int(self.box_in_bg[0])) + "," + str(int(self.box_in_bg[1])) + "," +
                    str(int(self.box_in_bg[2])) + "," + str(int(self.box_in_bg[3])) + "," + str(self.type))

    def random_center(self):
        # 随机生成贴合坐标，left-up
        x_random = random.randint(0, 1100)
        # x_random = random.randint(1150, 1200)
        y_random = random.randint(0, 800)
        # y_random = random.randint(800, 900)
        self.center = (x_random, y_random)
        return self.trans_coord()

    def stack_up(self):
        """
        图片贴合并进行保存

        :param src_img: 需要贴合的图片
        :param bg_img: 背景图
        :return:
        """
        # global n
        # dst.paste(src_img, (100, 400), src_img)
        # self.bg_prod = self.bg.copy()  # 进行gama处理的时候已经赋值self.bg_prod,此处可以关闭
        # self.bg_prod.paste(self.rotated_img, self.center, self.rotated_img)
        Aug.class_bg.paste(self.rotated_img, self.center, self.rotated_img)
        # 生成文件名，尽量保证唯一
        # t = datetime.strftime(datetime.today(), "%Y%m%d%H%M%S")
        # self.bg_prod.save(self.img_path + "/" + t + str(n) + ".jpg")
        # Aug.class_bg.save(self.img_path + "/" + t + str(n) + ".jpg")
        # print("已保存类别{}，{}张!".format(self.type, n+1))
        # self.file_name = t + str(n)
        # 生成txt标注文件
        # self.img_txt()
        # n += 1

    def img_gama(self):
        """
        对背景和bottle进行随机亮度调整，然后进行贴合


        :return:
        """
        self.img_augm()
        # self.bg_prod = self.bg_prod
        # self.bg_prod = self.bg_arr
        # 对bg进行处理，随机生成0.6-1.5之间的factor，调整bg的亮度
        self.bg_prod = ImageEnhance.Brightness(self.bg_prod).enhance(round(random.uniform(0.6, 1.5), 1))
        # 对瓶子进行处理
        self.rotated_img = ImageEnhance.Brightness(self.rotated_img).enhance(round(random.uniform(0.5, 1.2), 1))


def aug_iou(points, type_labs, _aug_obj_list):
    coordinates = np.array(points)
    # type_labs_cp = type_labs.copy()
    rm_list = []
    for i in range(len(points) - 1):
        if list(coordinates[i]) in rm_list:
            continue
        # print(coordinates[i])
        b1_x1 = coordinates[i][0]
        b1_y1 = coordinates[i][1]
        b1_x2 = coordinates[i][2]
        b1_y2 = coordinates[i][3]
        cp_coords = coordinates[i + 1:]
        # cp_labels = type_labs_cp[i + 1:]
        for m in range(len(cp_coords)):
            if list(cp_coords[m]) in rm_list:
                continue
            b2_x1 = cp_coords[m][0]
            b2_y1 = cp_coords[m][1]
            b2_x2 = cp_coords[m][2]
            b2_y2 = cp_coords[m][3]
            # 排除大框包裹小框的情况，删除小框
            if b1_x1 <= b2_x1 and b1_x2 >= b2_x2 and b1_y1 <= b2_y1 and b1_y2 >= b2_y2:
                points.remove(list(cp_coords[m]))
                # type_labs.remove(cp_labels[m])
                type_labs[i + m + 1] = "e"
                _aug_obj_list[i + m + 1] = "e"
                rm_list.append(list(cp_coords[m]))
                continue
            if b1_x1 >= b2_x1 and b1_x2 <= b2_x2 and b1_y1 >= b2_y1 and b1_y2 <= b2_y2:
                # rm_box = random.sample([coordinates[i], coordinates[i+1:][m]], 1)
                # rm_boxes.add(coordinates[i+1:][m])
                # try:
                points.remove(list(coordinates[i]))
                # type_labs.remove(type_labs_cp[i])
                # 所有需要删除的bbox置为"e",方便后面进行处理
                type_labs[i] = "e"
                _aug_obj_list[i] = "e"
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
            print("IOU is %0.2f" % iou)
            # 设置条件，如果iou超过阈值，只保留一个
            if iou >= 0.3:
                # rm_boxes.add(coordinates[i+1:][m])
                # print(list(cp_coords[m]))
                # if list(coordinates[i]) in rm_list:
                #     break
                points.remove(list(coordinates[i]))
                # type_labs.remove(type_labs_cp[i])
                type_labs[i] = "e"
                _aug_obj_list[i] = "e"
                rm_list.append(list(coordinates[i]))
                break
    # 将type_labs中的"e"删除,用于处理多个相同标签的处理
    while True:
        if "e" not in type_labs:
            break
        type_labs.remove("e")
    while True:
        if "e" not in _aug_obj_list:
            break
        _aug_obj_list.remove("e")
    # print("points", points)
    return _aug_obj_list, points, type_labs


def main():
    """
    主程序

    :param img_num: 需要生成的图片数量
    :return:
    """
    aug = Aug(src, bg, btype, img_save_path, anno_save_path)
    # for _ in range(img_num):
    # if not angle % 2 == 0:
    #     continue
    # 旋转图片
    aug.rotate_src()
    # 获得旋转后物体的位置
    aug.get_obj_box()
    # 对旋转后的图片进行亮度调整
    aug.img_gama()
    # 随机生成贴合中心点坐标
    coord_in_bg = aug.random_center()
    aug_obj_list.append(aug)
    src_boxes.append(coord_in_bg)
    type_labels.append(aug.type)
    # TODO 贴合之前进行IOU筛出

    # 进行贴合，并保存物体在背景下的坐标
    # aug.stack_up()
    # 进行画框，测试时用。注意：如果开启显示，执行速度降低而且要保证你的内存足够大。生产使用，建议关闭
    # aug.draw_box()


if __name__ == '__main__':
    img_save_path = "dataset_13_test"
    anno_save_path = "anno_13_test"
    img_total_num = 1000
    # 生成的图片统一保存到dataset文件夹
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)
    if not os.path.exists(anno_save_path):
        os.mkdir(anno_save_path)
    # 将需要的物体存放列表
    # bottle_list = ["01", "02", "03", "04", "05", "06"]
    bottle_list = ["%02d" % (i + 1) for i in range(3)]  # 选取6张bottle透明背景图
    # bg_id = ["01", "02", "03"]
    bg_id = ["%02d" % (i + 1) for i in range(10)]  # 选取10张背景图
    bottle_num = len(bottle_list)
    bottle_num_per_type = 10  # 指定所使用的每个类别的瓶子数量
    bottle_types = ["01", "02", "03"]
    # src_path = []
    # for m in range(len(bottle_num_per_type)):
    #     # 生成src集合
    #     src_path += ["images/bottle_{}_{}.png".format(i, random.sample(bottle_list, 1))
    #     for i in range(len(bottle_types))]
    for i in range(1):
        # bottle need to paste, 瓶子种类，遍历，瓶子序号随机取
        # src_path = ["images/bottle_{}_{}.png".format(bottle_types[i], random.sample(bottle_list, 1))
        #             for i in range(len(bottle_types))]
        src_boxes = []  # 保存透明背景的bottle
        type_labels = []  # 保存标签
        aug_obj_list = []   # 保存aug的实例对象
        for j in range(len(bottle_types)):
            src = Image.open("images/bottle_{}_{}.png".format(bottle_types[j], random.sample(bottle_list, 1)[0])).convert("RGBA")
            btype = bottle_types[j]
            bg_path = "images/bg.jpg"
            bg = Image.open(bg_path)
            main()
        # 生成符合要求的boxes
        # assert len(src_boxes) == len(type_labels), "生成src_boxes和type_labels不匹配！"
        aug_obj_list, src_boxes, type_labels = aug_iou(src_boxes, type_labels, aug_obj_list)
        # 生成文件名的时间戳
        t = datetime.strftime(datetime.today(), "%Y%m%d%H%M%S")
        file_name = t + str(n)
        f = open(anno_save_path + "/" + file_name + ".txt", "w")
        for obj in zip(aug_obj_list, src_boxes, type_labels):
            # obj --> (aug_obj, box, type)
            self = obj[0]
            self.stack_up()
            f.write(str(int(self.box_in_bg[0])) + "," + str(int(self.box_in_bg[1])) + "," +
                    str(int(self.box_in_bg[2])) + "," + str(int(self.box_in_bg[3])) + "," + str(self.type) + "\n")
        Aug.class_bg.save(img_save_path + "/" + file_name + ".jpg")
        f.close()
        print("图片已保存{}张".format(n + 1))
        draw = ImageDraw.Draw(Aug.class_bg)
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=40)
        for x in zip(src_boxes, type_labels):
            draw.rectangle(x[0], outline=(random.randrange(256), random.randrange(256), random.randrange(256)), width=2)
            draw.text((x[0][0], x[0][1]), text=x[1], fill=(0, 0, 0), font=font)
        del draw
        Aug.class_bg.show()
        n += 1
