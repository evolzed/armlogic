from src.Vision.imageProcess.bgLearn import Bglearn
import cv2
import numpy as np
from src.Vision.camera import Camera
from timeit import default_timer as timer

class ImageTrack:
    def __init__(self):
        self.MAX_CORNERS = 50
        self.win_size = 10

    def eDistance(self, p1, p2):
        #function description:
        # get Euclidean distance  between point p1 and p2
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def correctAngle(self,rbox):
        print("rbox", rbox)
        print("rboxtype", type(rbox))
        # mrbox=np.array(rbox)
        w = self.eDistance(rbox[0], rbox[1])
        h = self.eDistance(rbox[1], rbox[2])
        print("w", w)
        print("h", h)
        # 钝角 内积小于0
        xAxisVector = np.array([[0], [1]])
        angle = 0
        # find the long side of rotate rect
        if w > h:
            # find the low point the vector is from low to high
            v = np.zeros(rbox[0].shape, dtype=rbox.dtype)

            v = rbox[1] - rbox[0]
            print("v:", v)
            print("vdet", np.dot(v, xAxisVector)[0])
            # 内积大于0 是锐角 小于0是钝角
            if np.dot(v, xAxisVector)[0] > 0:
                angle = - angle
            if np.dot(v, xAxisVector)[0] < 0:
                angle = - angle + 90
            if angle == 0:
                angle = 90

        # find the long side of rotate rect
        if h > w:
            # find the low point the vector is from low to high
            v = np.zeros(rbox[0].shape, dtype=rbox.dtype)
            v = rbox[1] - rbox[2]
            print("v:", v)
            print("vdet", np.dot(v, xAxisVector)[0])
            # 内积大于0 是锐角 小于0是钝角
            if np.dot(v, xAxisVector)[0] > 0:
                angle = - angle
            if np.dot(v, xAxisVector)[0] < 0:
                angle = - angle + 90
            if angle == 0:
                angle = 90

        #while above angle is 0--180
        #then we will refact the angle to -90----90

        if 0 <= angle <= 90:
            angle = angle
        if 0 < angle < 180:
            angle = angle - 180
        return angle

    def getBottlePose(self, frameOrg0, bgMask, dataDict):
        # function description:
        # get Bottle Pose info,include bottle rotate angle and the diameter of bottle
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
        dataDict:
            add the dataDict bottle rotate angle from belt move direction,-90--0 degree,
            and the diameter of bottle
        -------

        Examples
        --------

        测试方法，该方法不能单独测试，是随着main_test方法一同测试的，运行main_test方法即可测试，观察pos窗口中
        的瓶子有没有一个紫色可以旋转的框，跟着瓶子的旋转角度在动，然后观察Log,查看dataDict的‘box’条目下的最后
        两个数字，一个是角度，一个是直径，是否正确，例如
        'box': [['Transparent bottle', 0.3777146, 568, 378, 679, 465, 0.0, 60.19998931884765]
        0.0是角度  60.19998931884765是直径


        """
        # init a morph kernel
        prev_time = timer()
        kernel19 = np.ones((19, 19), np.uint8)
        #get the frame from cam
        frameOrg = frameOrg0.copy()
        if "box" in dataDict:
            for i in range(len(dataDict["box"])):
                #get the box vertex
                left = dataDict["box"][i][2]
                top = dataDict["box"][i][3]
                right = dataDict["box"][i][4]
                bottom = dataDict["box"][i][5]
                rectTop = np.array([left, top])
                rectBottle = (right-left, bottom - top)
                # get the BOX ROI from frame
                roiImg = frameOrg[left:right, top:bottom]

                #externd region
                # right += 40
                # left -= 40
                # bottom += 40
                # top -= 40
                # get the center of box
                centerX = (right+left/2)
                centerY = (top + bottom / 2)

                #prevent beyond the pic limit
                if (right-left > 1) and (bottom - top > 1) and left > 0 and right < bgMask.shape[1]\
                        and bottom < bgMask.shape[0]and top > 0:
                    #display the box region of bgMask
                    #cv2.imshow("box"+str(i), bgMask[top:bottom, left:right])
                    roi = bgMask[top:bottom, left:right]
                    # erode and find contour  prevent too fat
                    roi = cv2.erode(roi, kernel19)  # eclipice
                    if cv2.__version__.startswith("3"):
                        _, contours, hierarchy = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    elif cv2.__version__.startswith("4"):
                        contours, hierarchy = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    #cv2.drawContours(self.show, contours, -1, (0, 255, 0), 3)
                    contourLen = len(contours)
                    # print(contourLen)
                    momentList = []
                    pointList = []
                    #print("contour", contourLen)
                    # choose the most propreate contour
                    if contourLen > 0:
                        index = 0
                        for j in range(contourLen):
                            # contourM = cv2.moments(contours[j])
                            # contourCenterGx = int(contourM['m10'] / contourM['m00'])
                            # contourCenterGy = int(contourM['m01'] / contourM['m00'])
                            contourArea = cv2.contourArea(contours[j])
                            #print("contourArea:", contourArea)
                           #  print("pixNum:", pixNum)
                            #print("total:", (right - left) * (bottom - top))
                            if (contourArea/((right-left)*(bottom - top))) > 0.2:
                            #if (abs(contourCenterGx-centerX)+abs(contourCenterGy-centerY)) < abs(left-right)/4:
                                index = j

                        # https: // blog.csdn.net / lanyuelvyun / article / details / 76614872
                        # find the rotateRect and get the angle and diameter
                        rotateRect = cv2.minAreaRect(contours[index])
                        angle = rotateRect[2]
                        diameter = min(rotateRect[1][0], rotateRect[1][1])*0.6
                        rbox = cv2.boxPoints(rotateRect)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)

                        angle = self.correctAngle(rbox)

                        """
                        #in rbox  out correct angle
                        print("rbox", rbox)
                        print("rboxtype", type(rbox))
                        #mrbox=np.array(rbox)
                        w = self.eDistance(rbox[0], rbox[1])
                        h = self.eDistance(rbox[1], rbox[2])
                        print("w", w)
                        print("h", h)
                        #钝角 内积小于0
                        xAxisVector = np.array([[0], [1]])
                        #find the long side of rotate rect
                        if w > h:
                            #find the low point the vector is from low to high
                            v = np.zeros(rbox[0].shape, dtype=rbox.dtype)

                            v = rbox[1] - rbox[0]
                            print("v:", v)
                            print("vdet", np.dot(v, xAxisVector)[0])
                            # 内积大于0 是锐角 小于0是钝角
                            if np.dot(v, xAxisVector)[0] > 0:
                                angle = - angle
                            if np.dot(v, xAxisVector)[0] < 0:
                                angle = - angle + 90
                            if angle == 0:
                                angle = 90

                            # if rbox[0][1] > rbox[1][1]:
                            #     v = rbox[0] - rbox[1]
                            #     print("v:",v)
                            #     print("vdet:", np.dot(v, xAxisVector)[0])
                            #     # 内积大于0 是锐角 小于0是钝角
                            #     if np.dot(v, xAxisVector)[0] > 0:
                            #         angle = - angle
                            #     if np.dot(v, xAxisVector)[0] < 0:
                            #         angle = - angle + 90
                            #
                            # if rbox[0][1] < rbox[1][1]:
                            #     v = rbox[1] - rbox[0]
                            #     print("v:", v)
                            #     print("vdet", np.dot(v, xAxisVector)[0])
                            #     # 内积大于0 是锐角 小于0是钝角
                            #     if np.dot(v, xAxisVector)[0] > 0:
                            #         angle = - angle
                            #     if np.dot(v, xAxisVector)[0] < 0:
                            #         angle = - angle + 90

                        # find the long side of rotate rect
                        if h > w:
                            #find the low point the vector is from low to high
                            v = np.zeros(rbox[0].shape, dtype=rbox.dtype)
                            v = rbox[1] - rbox[2]
                            print("v:", v)
                            print("vdet", np.dot(v, xAxisVector)[0])
                            # 内积大于0 是锐角 小于0是钝角
                            if np.dot(v, xAxisVector)[0] > 0:
                                angle = - angle
                            if np.dot(v, xAxisVector)[0] < 0:
                                angle = - angle + 90
                            if angle == 0:
                                angle = 90

                            # if rbox[1][1] > rbox[2][1]:
                            #     v = rbox[1] - rbox[2]
                            #     # 内积大于0 是锐角 小于0是钝角
                            #     if np.dot(v, xAxisVector)[0] > 0:
                            #         angle = - angle
                            #     if np.dot(v, xAxisVector)[0] < 0:
                            #         angle = - angle + 90
                            #
                            # if rbox[1][1] < rbox[2][1]:
                            #     v = rbox[2] - rbox[1]
                            #     # 内积大于0 是锐角 小于0是钝角
                            #     if np.dot(v, xAxisVector)[0] > 0:
                            #         angle = - angle
                            #     if np.dot(v, xAxisVector)[0] < 0:
                            #         angle = - angle + 90

                        # rbox[0] rbox[1]
                        # rbox[1] rbox[2]
                        """

                        rbox = rbox + rectTop
                        rbox = np.int0(rbox)
                        # 画出来
                        cv2.drawContours(frameOrg, [rbox], 0, (255, 0, 255), 1)
                        cv2.imshow("pos",frameOrg)
                        #store angle and diameter to the dataDict
                        dataDict["box"][i][6] = angle
                        dataDict["box"][i][7] = diameter
        curr_time = timer()
        exec_time = curr_time - prev_time  # 计算图像识别的执行时间
        dataDict["getPosTimeCost"] = exec_time
        return dataDict

    # analyse every point
    def analyseTrackPoint(self, good_new, good_old):
        # good_new = cornersB[st == 1]
        # good_old = cornersA[st == 1]

        #good_new -good_old
        dis = self.eDistance(good_new, good_old)
        return dis

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

    def lkLightflow_track(self, featureimg,  secondimg_orig, mask):
        # function description:
        # LK algorithm for track,input the featureimg  and  secondimg_orig, detetced the feature point in featureimg,
        # and then track the point of featureimg to get the corresponding point of secondimg_orig

        #we pass the previous frame, previous points and next frame.It returns next points along with some status numbers
        # which has a value of 1 if next point is found,

        """
        Parameters
        --------------
        I: featureimg
           secondimg_orig
          dataDict

        Returns
        -------

        Examples
        --------
        """
        #params for find good corners
        feature_params = dict(maxCorners=30,
                              qualityLevel=0.3,
                              minDistance=7,     #min distance between corners
                              blockSize=7)     #winsize of corner
        # params for lk track
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        #generate array of random colors
        #######color = np.random.randint(0, 255, (100, 3))
        #drawimg = featureimg.copy()
        #for drawing
        #drawimg = secondimg_orig.copy()
        #drawimg2 = secondimg_orig.copy()
        #change to gray
        featureimg = cv2.cvtColor(featureimg,  cv2.COLOR_BGR2GRAY)
        secondimg = cv2.cvtColor(secondimg_orig, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("res0", featureimg)
        # cv2.imshow("re1s", secondimg)
        corner_count = self.MAX_CORNERS
        #find the good corners for track
        cornersA = cv2.goodFeaturesToTrack(featureimg, mask=None, **feature_params)

       # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
      #  cv2.cornerSubPix(featureimg, cornersA, (self.win_size, self.win_size), (-1, -1), criteria)

        #corners_cnt = cornersA.size().height
        #get matrix row num

       # corners_cnt = cornersA.shape[0]
        #pyramid1 = cv2.buildOpticalFlowPyramid(featureimg, (self.win_size, self.win_size), 3)
        #pyramid2 = cv2.buildOpticalFlowPyramid(secondimg,  (self.win_size, self.win_size), 3)
        #print("corners_cnt", corners_cnt)
        cornersB = np.zeros(shape=cornersA.shape, dtype=cornersA.dtype)
        #light flow,pass the featureimg  and secondimg.It returns next points along with some st numbers
        # which has a value of 1 if next point is found,
        cornersB, st, err = cv2.calcOpticalFlowPyrLK(featureimg, secondimg, cornersA, None, **lk_params)
        #find the point that concerned to be tracked
        good_new = cornersB[st == 1]
        good_old = cornersA[st == 1]

        distance = self.analyseTrackPoint(good_new, good_old)
        print("distance", distance)

        #mask = np.zeros_like(drawimg)
        img = np.zeros_like(mask)
        #drawimg = np.zeros_like(mask) # mask every pic excetp the light flow angle
        drawimg = secondimg_orig.copy()
        #mask = drawimg.copy()
        #draw line between the tracked corners of qpre frame and current frameq
        for i, (new, old) in enumerate(zip(good_new, good_old)):  #fold and enumerate with i
            a, b = new.ravel()  #unfold
            c, d = old.ravel()
            # mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 1)
            mask = cv2.line(mask, (a, b), (c, d), (0, 0, 255), 1)

            if self.eDistance(np.array(list((a, b))), np.array(list((c, d)))) > 10:
                # drawimg = cv2.circle(drawimg, (a, b), 5, color[i].tolist(), -1)
                drawimg = cv2.circle(drawimg, (a, b), 5, (0, 0, 255), -1)



            img = cv2.add(drawimg, mask)

        """
        for i in range(corners_cnt):
            p0 = (cornersA[i, 0, 0], cornersA[i, 0, 1])#original  red
            p1 = (cornersB[i, 0, 0], cornersB[i, 0, 1])#now
            cv2.circle(drawimg, p0, 2, (0, 0, 255), -1)
            cv2.circle(drawimg, p1, 2, (0, 255, 0), -1)

            cv2.line(drawimg, p0, p1, (0, 255, 255), 1)
        """
        return good_new, good_old, img

if __name__ == "__main__":

    # a = cv2.imread("E:\\EvolzedArmlogic\\armlogic\\src\\Vision\\imageProcess\\1.jpg")
    # b = cv2.imread("E:\\EvolzedArmlogic\\armlogic\\src\\Vision\\imageProcess\\2.jpg")
    #print(a.type())
    #print(b.type())
    # wait for test multi bottles track
    obj = ImageTrack()
    try:
        cam = Camera()

        preFrame, nFrameNum, t = cam.getImage()
        #preFrame = np.zeros_like(frame)
        mask = preFrame.copy()
        timeStart = timer()
        timeCnt = 0
        while 1:
            frame, nFrameNum, t = cam.getImage()
            # camra fault tolerant
            # mem addr use is
            if frame is None:
                continue
            #drawimg = mask.copy()
            trakSart= timer()
            good_new, good_old, drawimg = obj.lkLightflow_track(preFrame, frame, mask)
            print("good_new.shape:", good_new.shape)
            print("good_old.shape:", good_old.shape)
            cv2.imshow("res", drawimg)
            #copy the current frame as preFrame for next use
            preFrame = frame.copy()
            #cv2.waitKey(10)
            ellapseTime = timer() - timeStart
            trackCostTime = timer() - trakSart
            print("trackCostTime:", trackCostTime)
            # update the timeStart and clear tqqhe mask
            if ellapseTime > 3:
                timeStart = timer()
                mask = np.zeros_like(preFrame)  #only detect the motion object

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cam.destroy()
                break
    except Exception as e:
        print(e)
        cam.destroy()



    # p1 = np.array([0, 0])
    # p2 = np.array([2, 2])
    # print("distance:", obj.eDistance(p1, p2))
    # cv2.imshow("res", drawimg)
    # cv2.waitKey()
    # print("sz:", corners_cnt)






