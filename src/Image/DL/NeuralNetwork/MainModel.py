from .YOLONano import YOLONano


class YOLONano(YOLONano):
    """YOLO Nano主模型
        用于做目标检测，和目标分类任务
        输入数据为张量，输出向量长度为：5+类别数
        具体实现在YOLONano.py文件中
    """

    def __init__(self):
        super().__init__()
