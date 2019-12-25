from src.Image.camera import Camera
from src.Image.image import Image
from src.Image.yolo.Yolo import YOLO

dstList = []  # use for store infomation of per box

def getBeltSpeed(dataDict):
    # 功能需求描述:
    # 获取传送带速度，以 (Vx ,Vy. Vz)的格式返回，此格式可以代表速度的大小和方向

    # 实现方法：
    # 首先使用两个连续帧作为一组，得到它们的dataDict，找到两帧中同一类型的瓶子，而且该类瓶子在每帧中只有一个，

    # 获取第一帧中该瓶子的矩形框的中心点坐标（x1,y1），并将该点根据传送带和水平方向的夹角，
    # 映射到传送带方向上，得到（x1',y1'）
    # 获取第二帧中该瓶子的矩形框的中心点坐标（x2,y2），并将该点根据传送带和水平方向的夹角，
    # 映射到传送带方向上，得到（x2',y2'）
    # 计算出来（x1',y1'）和（x2',y2'）的欧氏距离，除以两帧之间的时间，得到速度，并分解到X,Y,Z 方向上
    # 得到(Vx ,Vy. Vz)，其中Vz默认是0
    # 然后再继续取10组连续帧，重复上述计算，每组得到得到一个(Vx ,Vy. Vz)，将这些(Vx ,Vy. Vz)都存入
    # 一个(Vx ,Vy. Vz)数组
    # 对这个(Vx ,Vy. Vz)数组，剔除过大和过小的异常数据，可以使用np.percentile方法，然后对剩余数据求平均获得
    # 最终（Vx ,Vy. Vz)
    """
    Parameters
    --------------
    I: input:
      dataDict

    Returns
    bottleDetail:
        mainly include bottle rotate angle from belt move direction,0--180 degree,
        and the diameter of bottle
    -------

    Examples
    --------
    """

    # try get info of box in dataDict
    boxInfo = dataDict.get("box", 0)
    nFrame = dataDict["nFrame"]
    frameTime = dataDict["frameTime"]
    dstList.append(nFrame)
    dstList.append(frameTime)

    if boxInfo == 0:
        print("未检测到运动物体！")
        return None
    # get the first object which conditional more than 80%
    for i in boxInfo:
        if float(i[1]) > 0.8:
            dstList.append(i[0])  # 瓶子类别
            # 换算成中心坐标
            centerX = (dstList.append(i[2]) + dstList.append(i[4])) / 2
            centerY = (dstList.append(i[3]) + dstList.append(i[5])) / 2
            dstList.append(centerX)
            dstList.append(centerY)
            # test just one bottle
            break
    if not dstList:
        print("未检测到运动物体！")
        return None


if __name__ == '__main__':
    cam = Camera()
    yolo = YOLO()
    img = Image(cam, yolo)
    dataDict = img.detectSerialImage(cam)
    getBeltSpeed(dataDict)


