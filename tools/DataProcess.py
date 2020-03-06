import os
import sys
import time
import json
import PIL.Image as pimg
import keras

sys.path.append("../")
from src.Vision.yolo.Yolo import YOLO


class DataProcess():
    """
        数据处理
    """

    def __init__(self):
        """
            初始化信息：姓名、数据源路径、输出路径、处理时间
        """
        self.name = input("input name")
        self.inPath = input("input data path")
        while not os.path.isdir(self.inPath):
            self.inPath=input("Reenter data path")
        # '''测试用'''
        # self.inPath = r'F:\XSCX\Samples\validation\dtprcs'

        self.dataTime = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        self.yolo = YOLO()

    def Processing(self):
        """Data Process

        数据重命名，信息写入硬盘
        return: None
        """
        # 获取文件列表
        fileNameList = os.listdir(self.inPath)
        id = 0
        # 遍历目录中的文件
        for name in fileNameList:

            # 判断文件是否为图像文件
            if name.split('.')[-1] not in ['jpg', 'png', 'jpeg', 'bmp']:
                continue

            # 获取数据的完整路径
            file = os.path.join(self.inPath, name)

            # 判断路径是否为文件，不是文件则跳过
            if not os.path.isfile(file):
                continue

            # 存储图片尺寸和通道信息
            size = {}
            # 调用YOLO
            with pimg.open(file) as img:
                size['width'], size['height'], size['depth'] = *img.size, len(img.mode)
                result = self.yolo.detectImage(img)

            # 数据处理完成后对文件进行重命名
            newName = '{}{}'.format(self.dataTime, id)
            os.rename(os.path.join(self.inPath, name), os.path.join(self.inPath, '%s.jpg' % newName))  # 文件重命名

            self.SaveAnnotation2JSON(newName, result['box'], size)  # 保存标注信息
            id += 1

    def SaveAnnotation2JSON(self, fname, outputs, size):
        '''
        保存标注信息到JSON文件
        :param str: 要保存的标注信息
        :return:0(成功) or 1(失败)
        '''
        ann = {"path": "", "outputs": {}, "size": {}}  # 标注信息
        obj = []  # 目标列表
        target = {"name": "", "bandbox": {}}  # 目标信息
        bandbox = {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}  # 目标位置

        # 进行数据整合
        if outputs != []:
            for box in outputs:
                bandbox["xmin"],bandbox["ymin"],bandbox["xmax"],bandbox["ymax"]=int(box[2]),int(box[3]),int(box[4]),int(box[5])
                target["name"],target["bandbox"] =box[0], bandbox
                obj.append(target)

        ann["outputs"]["object"] = obj
        ann["size"] = size
        ann["path"] = '%s.jpg' % fname

        # 保存数据到文件
        dir=os.path.join(self.inPath, 'outputs')
        fp = os.path.join(dir, '%s.json' % fname)  # 文件保存路径
        if not os.path.exists(dir):
            os.mkdir(dir)
        try:
            with open(fp, 'w', encoding='utf-8') as fd:
                json.dump(ann, fd)
        except:
            return 1
        return 0


if __name__ == '__main__':
    process = DataProcess()
    process.Processing()
