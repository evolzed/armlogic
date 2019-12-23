from src.Image.imageProcess.bgLearn import Bglearn
import cv2
import numpy as np

class ImageTrack:
    def __init__(self):
        self.MAX_CORNERS = 500
        self.win_size = 10

    def getBottleDetail(self, dataDict):
        # function description:
        # get Bottle Detail info,include bottle rotate angle and the diameter of bottle
        #implementation detail:
        # first use the dataDict to get the bottle bondingbox  cordinates,
        #and then find the bottle contour in the region of bottle bondingbox
        #and then get the rotated box of bottle contour
        #finally,get the rotate angle  and width of  rotated box

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


    def getBeltSpeed(self, dataDict):
        # 功能需求描述:
        # 获取传送带速度，以 (Vx ,Vy. Vz)的格式返回，此格式可以代表速度的大小和方向

        # 实现方法：
        # 首先使用两个连续帧作为一组，得到它们的dataDict，找到两帧中同一类型的瓶子，而且该类瓶子在每帧中只有一个，
        # 获取第一帧中该瓶子的矩形框的中心点坐标（x1,y1），并将该点根据传送带和水平方向的夹角，
        # 映射到传送带方向上，得到（x1',y1'）
        # 获取第二帧中该瓶子的矩形框的中心点坐标（x2,y2），并将该点根据传送带和水平方向的夹角，
        # 映射到传送带方向上，得到（x2',y2'）
        #计算出来（x1',y1'）和（x2',y2'）的欧氏距离，除以两帧之间的时间，得到速度，并分解到X,Y,Z 方向上
        #得到(Vx ,Vy. Vz)，其中Vz默认是0
        #然后再继续取10组连续帧，重复上述计算，每组得到得到一个(Vx ,Vy. Vz)，将这些(Vx ,Vy. Vz)都存入
        #一个(Vx ,Vy. Vz)数组
        #对这个(Vx ,Vy. Vz)数组，剔除过大和过小的异常数据，可以使用np.percentile方法，然后对剩余数据求平均获得
        #最终（Vx ,Vy. Vz)
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

    def LKlightflow_track(self,featureimg,secondimg_orig):
        img_sz = featureimg.shape
        win_size = 10
        drawimg = featureimg.copy()
        drawimg2 = secondimg_orig.copy()
        secondimg =secondimg_orig.copy()
        featureimg = cv2.cvtColor(featureimg,  cv2.COLOR_BGR2GRAY )
        secondimg = cv2.cvtColor(secondimg_orig , cv2.COLOR_BGR2GRAY)
        corner_count = self.MAX_CORNERS
        cornersA = cv2.goodFeaturesToTrack(featureimg, corner_count, 0.01, 5.0)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        cv2.cornerSubPix(featureimg, cornersA, (self.win_size, self.win_size), (-1, -1), criteria)
        #corners_cnt = cornersA.size().height
        #get matrix row num
        corners_cnt = cornersA.shape[0]
        pyramid1 = cv2.buildOpticalFlowPyramid(featureimg, (self.win_size, self.win_size), 3)
        pyramid2 = cv2.buildOpticalFlowPyramid(secondimg,  (self.win_size, self.win_size), 3)
        cornersB = []
        cv2.calcOpticalFlowPyrLK(featureimg, secondimg, cornersA, cornersB)
        for i in range(corners_cnt):
            p0 = (cornersA[i, 0, 0], cornersA[i, 0, 1])#original
            p1 = (cornersB[i, 0, 0], cornersB[i, 0, 1])#now
            cv2.circle(drawimg, p0, 2, (0, 0, 255), -1)
            cv2.circle(drawimg, p1, 2, (0, 255, 0), -1)

            cv2.line(drawimg, p0, p1, (0, 255, 255), 1)

if __name__ == "__main__":
    a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    c = np.dot(a, b)
    print(c)

    obj = ImageTrack()
    a = np.array([[1, 2]])
    b = np.array([3, 4])
    corners_cnt = a.shape[1]
    #sz, drawimg = obj.LKlightflow_track(a, b)
    #c = np.zeros(shape=sz, dtype=float)
    print("sz:", corners_cnt)






