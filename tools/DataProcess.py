import os
import sys
import copy
import time
import json
import PIL.Image as pimg
import PIL.ImageDraw as draw
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
        self.name = input("input name：")
        self.inPath = input("input data path")
        while not os.path.isdir(self.inPath):
            self.inPath=input("Reenter data path")

        # self.inPath = r'F:\XSCX\Samples\validation\dtprcs'#测试用
        
        self.outPath=os.path.join(self.inPath, 'outputs')
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
        with open(os.path.join(self.outPath,'annotation.txt'),'w') as fd:
            fd.write('标注范围：%s-%s\n标注数量：%s\n标注时间：%s\n标注人员：%s\n'%(str(self.dataTime)+'0',str(self.dataTime)+str(id-1),id,self.dataTime,self.name))

    def SaveAnnotation2JSON(self, fname, outputs, size):
        '''
        保存标注信息到JSON文件
        :param str: 要保存的标注信息
        :return:0(成功) or 1(失败)
        '''
        ann = {"path": "", "outputs": {}, "size": {}}  # 标注信息
        obj = []  # 目标列表
        target = {"name": "", "bndbox": {}}  # 目标信息
        bndbox = {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}  # 目标位置

        # 进行数据整合
        if outputs != []:
            for box in outputs:
                bndbox["xmin"],bndbox["ymin"],bndbox["xmax"],bndbox["ymax"]=int(box[2]),int(box[3]),int(box[4]),int(box[5])
                target["name"],target["bndbox"] =box[0], bndbox
                obj.append(copy.deepcopy(target))

        ann["outputs"]["object"] = obj
        ann["size"] = size
        ann["path"] = '%s.jpg' % fname

        # 保存数据到文件↓
        # 文件保存路径
        fp = os.path.join(self.outPath, '%s.json' % fname)
        if not os.path.exists(self.outPath):
            os.mkdir(self.outPath)
        try:
            with open(fp, 'w', encoding='utf-8') as fd:
                json.dump(ann, fd)
        except:
            return 1
        return 0

    def FilterAnnotation(self):
        '''
        对标注信息进行过滤，记录不符合要求的图片
        :return: None
        '''
        # 获取文件列表
        fileList = os.listdir(self.outPath)
        # 统计未标注的文件
        un_anno = []
        for file in fileList:
            #遍历文件列表
            #判断文件类型
            if file.endswith(".json"):
                #打开json文件
                with open(os.path.join(self.outPath,file), "r") as f:
                    #解析json文件
                    dic = json.loads(f.read().encode("gbk").decode("utf8"))
                    #获取目标列表
                    outputs = dic.get("outputs")
                    #判断是否有目标
                    if outputs is None:
                        print("--None--" * 20, file)
                        continue
                    try:
                        obj = outputs["object"]
                    except Exception as e:
                        un_anno.append(file)
                        continue
                    #不包含后缀的文件名
                    name=file[:-5]
                    #打开对应的图像文件
                    with pimg.open(os.path.join(self.inPath,'%s.jpg'% name)) as img:
                        #创建画图对象
                        imgDraw=draw.Draw(img)
                        #获取所有的目标框
                        for bbox in obj:
                            bbox_class = bbox["name"]
                            bbox_xmin = bbox["bndbox"]["xmin"]
                            bbox_ymin = bbox["bndbox"]["ymin"]
                            bbox_xmax = bbox["bndbox"]["xmax"]
                            bbox_ymax = bbox["bndbox"]["ymax"]
                            imgDraw.rectangle((bbox_xmin,bbox_ymin,bbox_xmax,bbox_ymax))
                        #显示图像
                        img.show()
                        #标记数据是否需要重新标注
                        flag=input('是否需要重标,请输入 y 或 n：')
                        while not flag in ['y','n']:
                            flag = input('输入错误,请重新输入 y 或 n：')
                        if flag=='n':
                            continue
                        elif flag=='y':
                            #需要重新标注时，将文件名写入txt文档
                            with open(os.path.join(self.outPath, "reList.txt"),'a') as fd:
                                fd.write('%s.jpg\n'% name)


if __name__ == '__main__':
    process = DataProcess()
    process.Processing()
    print("processed")
    process.FilterAnnotation()

