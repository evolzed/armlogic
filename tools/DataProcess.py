import os
import time
import json

class DataProcess():
    """
        数据处理
    """

    def __init__(self):
        """
            初始化信息：姓名、数据源路径、输出路径、处理时间
        """
        self.name = input("name")
        self.inPath = input("input path")
        self.outPath = input("output path")
        self.dataTime =time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    def Processing(self):
        """Data Process

        数据重命名，信息写入硬盘
        return: None
        """
        #获取文件列表
        fileNameList=os.listdir(self.inPath)
        id=0
        #遍历目录中的文件
        for name in fileNameList:
            #获取数据的完整路径
            file=os.path.join(self.inPath,name)
            #判断路径是否为文件，不是文件则跳过
            if not os.path.isfile(file):
                continue

            '''
            yolo插入到此处
            '''
            anno=None

            #数据处理完成后对文件进行重命名
            newName='{}{}'.format(self.dataTime,id)
            os.rename(os.path.join(self.inPath,name),os.path.join(self.inPath,'%s.jpg'%newName))#文件重命名
            self.SaveAnnotation2JSON(newName,anno)#保存标注信息
            id+=1


    def SaveAnnotation2JSON(self,fname,anno):
        '''
        保存标注信息到JSON文件
        :param str: 要保存的标注信息
        :return:0(成功) or 1(失败)
        '''
        ann = {"path": "", "outputs": {}, "time_labeled": 0, "size": {}}#标注信息
        size = {"width": 0, "height": 0, "depth": 0}#图片尺寸和通道
        object = []#目标列表
        obj = {"name": "", "bandbox": {}}#目标信息
        bandbox = {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}#目标位置

        '''
        提取anno中的标注信息
        '''


        obj["bandbox"] = bandbox
        object.append(obj)
        ann["outputs"] = object
        ann["size"] = size
        fp=os.path.join('F:\XSCX','%s.json'%fname)#文件保存路径
        fd = open(fp, 'w', encoding='utf-8')
        json.dump(ann,fd)
        return 0



if __name__ == '__main__':
    process=DataProcess()
    # process.Processing()