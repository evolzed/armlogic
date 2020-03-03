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
            #判断路径是否为文件
            if not os.path.isfile(file):
                continue

            #数据处理完成后对文件进行重命名
            os.rename(os.path.join(self.inPath,name),os.path.join(self.inPath,))
            json.load()



if __name__ == '__main__':
    process=DataProcess()
    process.Processing()