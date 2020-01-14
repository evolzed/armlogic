import gc

import cv2
import numpy as np
from src.Vision.camera import Camera
from timeit import default_timer as timer

class ImgProc:
    def __init__(self, bgStudyNum):
        """
        :param bgStudyNum: how many pics captured for background study
        """
        # private attribute of class
        self.BG_STUDY_NUM = bgStudyNum
        # a list for store the pics captqured waited for study
        self.bgVector = np.zeros(shape=(self.BG_STUDY_NUM, 960, 1280, 3), dtype=np.float32)
        # average of frames
        self.IavgF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)
        # pre frame
        self.IprevF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)
        # temp frame for calculate
        self.Iscratch2 = np.zeros(shape=(960, 1280, 3), dtype=np.float32)
        # average difference of frames
        self.IdiffF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)
        # high threshold of background
        self.IhiF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)
        # low threshold of background
        self.IlowF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)
        # statistic the count of pics captured waited for study
        self.Icount = 0
        # statistic the Frame number Pre One Second
        # self.nFrameNumPreOneSec = 0

        # kernel for image morph process
        self.kernel5 = np.ones((5, 5), np.uint8)
        self.kernel7 = np.ones((7, 7), np.uint8)
        self.kernel13 = np.ones((13, 13), np.uint8)
        self.kernel19 = np.ones((19, 19), np.uint8)
        self.kernel25 = np.ones((25, 25), np.uint8)

        # for show
        self.show = np.zeros(shape=(960, 1280, 3), dtype=np.uint8)

        # time cost of one frame delete the background
        self.bgTimeCost = 0

        self.MAX_CORNERS = 50
        self.win_size = 10

    def avgBackground(self, I):
        """
        read background pic of I,and then calculate frame difference of current frame and pre frame: I and IprevF
        accumulate every frame difference Iscratch2 to sum of differences :IdiffF
        meanwhile,accumulate every frame I  to sum of frames :IavgF

        :param I: input  Mat type pic stream
        :return: None
        """
        cv2.accumulate(I, self.IavgF)
        # cv2.absdiff(I,IprevF, Iscratch2)
        self.Iscratch2 = cv2.absdiff(I, self.IprevF)
        cv2.accumulate(self.Iscratch2, self.IdiffF)

        # print("IavgF[100,100,0]:", self.IavgF[100, 100, 0])
        # print("IdiffF[100,100,0]:", self.IdiffF[100, 100, 0])
        self.Icount += 1.0
        self.IprevF = I.copy()

    def createModelsfromStats(self, scale):
        """
        calculate the average sum of frames to  IavgF
        calculate frame difference to IdiffF
        then  multiply the scale to the Idiiff,and add the Idiff to IavgF to get the IhiF,
        subtract  the Idiff from IavgF to get the IlowF
        now we get the background model IhiF and IlowF

        :param scale: gap of high threshold and low threshold of background model
        :return: None
        """
        # print("Icount", self.Icount)
        # Icount+=1
        self.IavgF = self.IavgF / self.Icount
        self.IdiffF = self.IdiffF / self.Icount
        # print("IavgF[100,100,0]:", self.IavgF[100, 100, 0])
        # print("IdiffF[100,100,0]:", self.IdiffF[100, 100, 0])
        self.IdiffF = cv2.add(self.IdiffF, 1.0)
        # cv2.imshow("avg", IavgF)
        # cv2.imshow("diff", IdiffF)
        # cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\a.jpg", self.IavgF)
        # cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\d.jpg", self.IdiffF)
        # cv2.add(IavgF,IdiffF, IhiF)

        self.IdiffF = self.IdiffF * scale
        # print("IdiffF[mod:", self.IdiffF[100, 100, 0])
        self.IhiF = cv2.add(self.IavgF, self.IdiffF)
        # cv2.subtract(IavgF, IdiffF, IlowF)
        self.IlowF = cv2.subtract(self.IavgF, self.IdiffF)
        # release the memory | 12.25 内存优化，显示释放内存
        del self.bgVector
        gc.collect()
        # cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\h.jpg", self.IhiF)
        # cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\l.jpg", self.IlowF)

    def studyBackgroundFromCam(self, cam):
        """
        get many pics for time interval of 60sec by cam and store the pics in  bgVector.
        then  call the avgBackground method

        :param cam: input camera object
        :return: None
        """
        try:
            # set the loop break condition over_flag,when pics waited for study is captured enough,the flag will be changed
            over_flag = 1
            pic_cnt = 0
            while (over_flag):
                # get image from camera
                frame, nFrameNum, t = cam.getImage()
                fin = np.float32(frame)
                # print("shape", fin.shape)
                # store the frame in list bgVector
                self.bgVector[pic_cnt] = fin
                # print("pic_cnt", pic_cnt)
                # wait about 200 milli seconds
                cv2.waitKey(200)
                pic_cnt += 1
                print("pic_cnt", pic_cnt)
                if (pic_cnt == self.BG_STUDY_NUM):
                    over_flag = 0

            # print("shapebg", self.bgVector.shape)
            for i in range(self.bgVector.shape[0]):
                # print("i", i)
                self.avgBackground(self.bgVector[i])
        except Exception as e:
            print(e)
            # when occur exception ,the camera will disconnect
            cam.destroy()

    def backgroundDiff(self, src0, dst):
        """
        when get pic frame from camera, use the backgroundDiff to  segment the frame pic and get a mask pic
        if the pic pixel value is higher than  high background threadhold  or lower than low background threadhold, the pixels
        will change to white,otherwise, it will cover to black.
        https://www.cnblogs.com/mrfri/p/8550328.html
        rectArray=np.zeros(shape=(1,4),dtype=float)

        :param src0: input cam pic waited for segment
        :param dst: temp store segment result of mask
        :return: rectArray:all boundingboxes of all bottles
                 dst:segment result of mask
        """
        rectArray = []
        src = np.float32(src0)
        # print("IlowF.shape", self.IlowF.shape)
        # print("IhiF.shape", self.IhiF.shape)
        # print("src.shape", src.shape)
        # print("dst.shape", dst.shape)

        # print("IlowF.tpye", self.IlowF.dtype)
        # print("IhiF.tpye", self.IhiF.dtype)
        # print("src.tpye", src.dtype)
        # print("dst.tpye", dst.dtype)

        # cv2.inRange(src, IlowF, IhiF, dst)
        # segment the src through IlowF and IhiF
        dst = cv2.inRange(src, self.IlowF, self.IhiF)
        # cv2.imshow("segment_debug", dst)

        # morph process the frame to clear the noise and highlight our object region
        # print("is this ok00?")
        dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, self.kernel7)
        # print("is this ok01?")
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, self.kernel7)
        # print("is this ok02?")

        tmp = 255 * np.ones(shape=dst.shape, dtype=dst.dtype)
        # np.zeros(shape=(960, 1280, 3), dtype=np.uint8)

        # inverse the  pixel value to make the mask
        dst = cv2.subtract(255, dst)
        # dst=tmp-dst
        # print("is this ok03?")
        # cv2.GaussianBlur(dst, dst, (19, 19), 3)

        # filter  and morph again and then find the bottle contours
        dst = cv2.GaussianBlur(dst, (19, 19), 3)
        # print("is this ok04?")
        dst = cv2.dilate(dst, self.kernel19)
        dst = cv2.dilate(dst, self.kernel19)
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, self.kernel13)  # eclipice
        dst = cv2.erode(dst, self.kernel19)  # eclipice
        # 解决cv2版本3.4.2和4.1.2兼容问题
        if cv2.__version__.startswith("3"):
            _, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        elif cv2.__version__.startswith("4"):
            contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(self.show, contours, -1, (0, 255, 0), 3)
        contourLen = len(contours)
        # print(contourLen)
        momentList = []
        pointList = []
        if contourLen > 0:  # range create a list of interger ,useful in loop
            for i in range(contourLen):
                # calculate all features of contours and draw rectangle of bounding box of contour
                contourM = cv2.moments(contours[i])  # every contour's  moment
                contourCenterGx = int(contourM['m10'] / contourM['m00'])  # 重心
                contourCenterGy = int(contourM['m01'] / contourM['m00'])
                contourArea = cv2.contourArea(contours[i])  # 面积
                contourhull = cv2.convexHull(contours[i])  # 凸包
                cv2.polylines(self.show, [contourhull], True, (500, 255, 0), 2)

                # https: // blog.csdn.net / lanyuelvyun / article / details / 76614872
                rotateRect = cv2.minAreaRect(contours[i])  # 旋转外接矩形
                angle = rotateRect[2]
                diameter = min(rotateRect[1][0], rotateRect[1][1])

                contourBndBox = cv2.boundingRect(contours[i])  # x,y,w,h  外接矩形
                # print("contourBndBox type", type(contourBndBox))
                x = contourBndBox[0]
                y = contourBndBox[1]
                w = contourBndBox[2]
                h = contourBndBox[3]
                img = cv2.rectangle(self.show, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画矩形
                rows, cols = src.shape[:2]  # shape 0 1 #得出原图的行 列 数
                [vx, vy, x, y] = cv2.fitLine(contours[i], cv2.DIST_L2, 0, 0.01, 0.01)  # 对轮廓进行多边形拟合
                lefty = int((x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)
                # print("pre final")
                # rectArray = np.append(rectArray, contourBndBox, axis=0)
                rectArray.append(contourBndBox)  # 存轮廓信息到数组中

                # print("final")
                # print("rectArray", rectArray)
        return rectArray, dst

    def delBg(self, src):
        """
        use the mask pic bgMask to make bit and operation to the cam frame to get a pic that del the bacground

        :param src: input cam pic waited for segment
        :return: rectArray:all boundingboxes of all bottles
                dst:segment result of mask
        """
        prev_time = timer()
        # simply output the frame that delete the background
        # dst = np.zeros(shape=(960, 1280, 3), dtype=np.uint8)  flexible the dst shape to adapt the src shape
        dst = np.zeros(shape=src.shape, dtype=src.dtype)
        resarray, bgMask = self.backgroundDiff(src, dst)  # 对src 帧 减去背景 结果放到dst，获得瓶子的框，和掩膜图像
        # bit and operation
        frame_delimite_bac = cv2.bitwise_and(src, src, mask=bgMask)  # 用掩膜图像和原图像做像素与操作，获得只有瓶子的图
        curr_time = timer()
        # calculate the cost time
        exec_time = curr_time - prev_time
        self.bgTimeCost = exec_time
        # print("Del background Cost time:", self.bgTimeCost)
        return frame_delimite_bac, bgMask, resarray

    def eDistance(self, p1, p2):
        """
        function description:
        get Euclidean distance  between point p1 and p2

        :param p1: point
        :param p2: point
        :return: distance
        """
        distance = np.sqrt(np.sum((p1 - p2) ** 2))
        return distance


    def correctAngle(self, rbox):
        """
        correct the angle to -90 to 90 for get Pose,

        :param rbox: input rotatebox
        :return: angle: angle that modified
        """
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

        # while above angle is 0--180
        # then we will refact the angle to -90----90

        if 0 <= angle <= 90:
            angle = angle
        if 0 < angle < 180:
            angle = angle - 180
        return angle


    def getBottlePose(self, frameOrg0, bgMask, dataDict):
        """
        function description:
        get Bottle Pose info,include bottle rotate angle and the diameter of bottle
        implementation detail:
        first use the dataDict to get the bottle bondingbox  cordinates,
        and then find the bottle contour in the region of bottle bondingbox
        and then get the rotated box of bottle contour
        finally,get the rotate angle  and width of  rotated box
           测试方法，该方法不能单独测试，是随着main_test方法一同测试的，运行main_test方法即可测试，观察pos窗口中
        的瓶子有没有一个紫色可以旋转的框，跟着瓶子的旋转角度在动，然后观察Log,查看dataDict的‘box’条目下的最后
        两个数字，一个是角度，一个是直径，是否正确，例如
        'box': [['Transparent bottle', 0.3777146, 568, 378, 679, 465, 0.0, 60.19998931884765]
        0.0是角度  60.19998931884765是直径

        :param frameOrg0: input frame
        :param bgMask:  the mask frame that generate from bglearn
        :param dataDict:  input dataDict
        :return: dataDict:output dataDict
        """
        # init a morph kernel
        prev_time = timer()
        kernel19 = np.ones((19, 19), np.uint8)
        # get the frame from cam
        frameOrg = frameOrg0.copy()
        if "box" in dataDict:
            for i in range(len(dataDict["box"])):
                # get the box vertex
                left = dataDict["box"][i][2]
                top = dataDict["box"][i][3]
                right = dataDict["box"][i][4]
                bottom = dataDict["box"][i][5]
                rectTop = np.array([left, top])
                rectBottle = (right - left, bottom - top)
                # get the BOX ROI from frame
                roiImg = frameOrg[left:right, top:bottom]

                # externd region
                # right += 40
                # left -= 40
                # bottom += 40
                # top -= 40
                # get the center of box
                centerX = (right + left / 2)
                centerY = (top + bottom / 2)

                # prevent beyond the pic limit
                if (right - left > 1) and (bottom - top > 1) and left > 0 and right < bgMask.shape[1] \
                        and bottom < bgMask.shape[0] and top > 0:
                    # display the box region of bgMask
                    # cv2.imshow("box"+str(i), bgMask[top:bottom, left:right])
                    roi = bgMask[top:bottom, left:right]
                    # erode and find contour  prevent too fat
                    roi = cv2.erode(roi, kernel19)  # eclipice
                    if cv2.__version__.startswith("3"):
                        _, contours, hierarchy = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    elif cv2.__version__.startswith("4"):
                        contours, hierarchy = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    # cv2.drawContours(self.show, contours, -1, (0, 255, 0), 3)
                    contourLen = len(contours)
                    # print(contourLen)
                    momentList = []
                    pointList = []
                    # print("contour", contourLen)
                    # choose the most propreate contour
                    if contourLen > 0:
                        index = 0
                        for j in range(contourLen):
                            # contourM = cv2.moments(contours[j])
                            # contourCenterGx = int(contourM['m10'] / contourM['m00'])
                            # contourCenterGy = int(contourM['m01'] / contourM['m00'])
                            contourArea = cv2.contourArea(contours[j])
                            # print("contourArea:", contourArea)
                            #  print("pixNum:", pixNum)
                            # print("total:", (right - left) * (bottom - top))
                            if (contourArea / ((right - left) * (bottom - top))) > 0.2:
                                # if (abs(contourCenterGx-centerX)+abs(contourCenterGy-centerY)) < abs(left-right)/4:
                                index = j

                        # https: // blog.csdn.net / lanyuelvyun / article / details / 76614872
                        # find the rotateRect and get the angle and diameter
                        rotateRect = cv2.minAreaRect(contours[index])
                        angle = rotateRect[2]
                        diameter = min(rotateRect[1][0], rotateRect[1][1]) * 0.6
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
                        cv2.imshow("pos", frameOrg)
                        # store angle and diameter to the dataDict
                        dataDict["box"][i][6] = angle
                        dataDict["box"][i][7] = diameter
        curr_time = timer()
        exec_time = curr_time - prev_time  # 计算图像识别的执行时间
        dataDict["getPosTimeCost"] = exec_time
        return dataDict


    # analyse every point
    def analyseTrackPoint(self, good_new, good_old, precisionThreshold):
        """
        analyse the track point to get the more precision point

        :param good_new: current frame point of track
        :param good_old: prev frame point of track
        :return: good_new0: more precision current frame point of track

                 good_old0: more precision prev frame point of track
        """
        # good_new = cornersB[st == 1]
        # good_old = cornersA[st == 1]

        # good_new -good_old
        #print("good_new shape", good_new.shape)
        #print("good_new shape[0]", good_new.shape[0])
        if np.isnan(good_new).sum() >0 or np.isnan(good_old).sum()>0:
            return good_new, good_old
        good_new0 = np.array([[0, 0]])
        good_old0 = np.array([[0, 0]])
        pointLen = good_new.shape[0]
        if pointLen == 0:
            return good_new, good_old
        disarray = np.array([])
        for i in range(pointLen):
            dis = self.eDistance(good_new[i], good_old[i])
            disarray = np.append(disarray, dis)
        #get the low 20% distance point,that is more precision points
        reduce = np.percentile(disarray, precisionThreshold, axis=0)
        reducearr = disarray[disarray <= reduce]
        index = np.where(disarray <= reduce)
        #format need
        index = index[0]
        print("index", index)
        # index_total = np.arrange(pointLen)
        # index = set(index.tolist())
        # index_total =set(index_total.tolist())
        # index_del =list(index_total.difference(index))
        print(np.array([good_new[0]]))
        for i in index:
            good_new0 = np.append(good_new0, np.array([good_new[i]]), axis=0)
            good_old0 = np.append(good_old0, np.array([good_old[i]]), axis=0)
        good_new0 = np.delete(good_new0, 0, axis=0)
        good_old0 = np.delete(good_old0, 0, axis=0)
        good_new0 = good_new0.astype(int)
        good_old0 = good_old0.astype(int)
        return good_new0, good_old0


    def getBeltSpeed(self, dataDict):
        """
        功能需求描述:
        获取传送带速度，以 (Vx ,Vy. Vz)的格式返回，此格式可以代表速度的大小和方向
        实现方法：
        首先使用两个连续帧作为一组，得到它们的dataDict，找到两帧中同一类型的瓶子，而且该类瓶子在每帧中只有一个，
        获取第一帧中该瓶子的矩形框的中心点坐标（x1,y1），并将该点根据传送带和水平方向的夹角，
        映射到传送带方向上，得到（x1',y1'）
        获取第二帧中该瓶子的矩形框的中心点坐标（x2,y2），并将该点根据传送带和水平方向的夹角，
        映射到传送带方向上，得到（x2',y2'）
        计算出来（x1',y1'）和（x2',y2'）的欧氏距离，除以两帧之间的时间，得到速度，并分解到X,Y,Z 方向上
        得到(Vx ,Vy. Vz)，其中Vz默认是0
        然后再继续取10组连续帧，重复上述计算，每组得到得到一个(Vx ,Vy. Vz)，将这些(Vx ,Vy. Vz)都存入
        一个(Vx ,Vy. Vz)数组
        对这个(Vx ,Vy. Vz)数组，剔除过大和过小的异常数据，可以使用np.percentile方法，然后对剩余数据求平均获得
        最终（Vx ,Vy. Vz)

        :param dataDict: bottle dictionary
        :return: bottleDetail:mainly include bottle rotate angle from belt move direction,0--180 degree,and the diameter of bottle
        """""


    def lkLightflow_track(self, featureimg, secondimg_orig, mask):
        """
        function description:
        LK algorithm for track,input the featureimg  and  secondimg_orig, detetced the feature point in featureimg,
        and then track the point of featureimg to get the corresponding point of secondimg_orig
        we pass the previous frame, previous points and next frame.It returns next points along with some status numbers
        which has a value of 1 if next point is found,

        :param featureimg:
        :param secondimg_orig:
        :param mask: mask for accumulate track lines,usually is preframe
        :return:  good_new:good tracked point of new frame,

                  good_old:old tracked point of new frame,

                  img :image for drawing
        """
        # params for find good corners
        feature_params = dict(maxCorners=30,
                              qualityLevel=0.3,
                              minDistance=7,  # min distance between corners
                              blockSize=7)  # winsize of corner
        # params for lk track
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # generate array of random colors
        #######color = np.random.randint(0, 255, (100, 3))
        # drawimg = featureimg.copy()
        # for drawing
        # drawimg = secondimg_orig.copy()
        # drawimg2 = secondimg_orig.copy()
        # change to gray
        featureimg = cv2.cvtColor(featureimg, cv2.COLOR_BGR2GRAY)
        secondimg = cv2.cvtColor(secondimg_orig, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("res0", featureimg)
        # cv2.imshow("re1s", secondimg)
        corner_count = self.MAX_CORNERS
        # find the good corners for track
        cornersA = cv2.goodFeaturesToTrack(featureimg, mask=None, **feature_params)
        if cornersA is None:
            return None, None, secondimg_orig
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        #  cv2.cornerSubPix(featureimg, cornersA, (self.win_size, self.win_size), (-1, -1), criteria)

        # corners_cnt = cornersA.size().height
        # get matrix row num

        # corners_cnt = cornersA.shape[0]
        # pyramid1 = cv2.buildOpticalFlowPyramid(featureimg, (self.win_size, self.win_size), 3)
        # pyramid2 = cv2.buildOpticalFlowPyramid(secondimg,  (self.win_size, self.win_size), 3)
        # print("corners_cnt", corners_cnt)
        #cornersB = np.zeros(shape=cornersA.shape, dtype=cornersA.dtype)  if corners A has no
        # light flow,pass the featureimg  and secondimg.It returns next points along with some st numbers
        # which has a value of 1 if next point is found,
        cornersB, st, err = cv2.calcOpticalFlowPyrLK(featureimg, secondimg, cornersA, None, **lk_params)
        # find the point that concerned to be tracked
        good_new = cornersB[st == 1]
        good_old = cornersA[st == 1]

        good_new, good_old = self.analyseTrackPoint(good_new, good_old, 30)



        #print("distancearr", distancearr)q
        #print("reduce", reduce)
        # mask = np.zeros_like(drawimg)
        img = np.zeros_like(mask)
        # drawimg = np.zeros_like(mask) # mask every pic excetp the light flow angle
        drawimg = secondimg_orig.copy()
        # mask = drawimg.copy()
        # draw line between the tracked corners of qpre frame and current frameq
        for i, (new, old) in enumerate(zip(good_new, good_old)):  # fold and enumerate with i
            a, b = new.ravel()  # unfold
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


def creatMaskFromROI(src, roi):
    """
    gernerate mask frame from roi,in roi the pixel will be set to 255,otherwise will set to 0
    :param src:
    :param roi:
    :return: mask
    """
    x = roi[0]
    y = roi[1]
    w = roi[2]
    h = roi[3]
    left = int(x - w/2)
    top = int(y + h/2)
    mask = np.zeros_like(src)
    mask[y:y+h, x:x+w, :] = 255
    #cv2.imshow("mask", mask)
    #mask need to be single channel
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask



if __name__ == "__main__":
    """
    a = cv2.imread("E:\\EvolzedArmlogic\\armlogic\\src\\Image\\imageProcess\\1.jpg")
    b = cv2.imread("E:\\EvolzedArmlogic\\armlogic\\src\\Image\\imageProcess\\5.jpg")
    #c = cv2.imread("E:\\EvolzedArmlogic\\armlogic\\src\\Image\\imageProcess\\5.jpg")


    #print(a.type())q
    #print(b.type())

    # wait for test multi bottles track

    mask = np.zeros_like(a)
    good_new, good_old, drawimg = obj.lkLightflow_track(a, b, mask)
    cv2.imshow("a", a)
    cv2.imshow("b", b)
    cv2.imshow("res", drawimg)
    cv2.waitKey()
    """
    obj = ImgProc(50)
    cam = Camera()
    obj.studyBackgroundFromCam(cam)
    obj.createModelsfromStats(6.0)

    try:
        # cam = Camera()
        preFrame, nFrameNum, t = cam.getImage()

        preframeDelBg, bgmask, resarray= obj.delBg(preFrame)


        #preFrame = np.zeros_like(frame)
        mask = preFrame.copy()

        premask = preframeDelBg.copy()
        timeStart = timer()
        timeCnt = 0
        while 1:
            frame, nFrameNum, t = cam.getImage()
            # camra fault tolerant
            # mem addr use is
            if frame is None:
                continue



            obj.show = frame.copy()
            # get fps of cam output
            fps = cam.getCamFps(nFrameNum)
            # use the background model to del the bacground of  frame

            frameDelBg, bgmask, resarray = obj.delBg(frame)

            #print(len(resarray))

            if resarray is not None:
                print("len：", len(resarray))
                for i in range(len(resarray)):
                    #print("resarray[i]：", resarray[i])
                    maskSingle = creatMaskFromROI(frameDelBg, resarray[i])
                    show = cv2.bitwise_and(preframeDelBg, preframeDelBg, mask=maskSingle)
                    #cv2.imshow(str(i), show)

            # put text on frame to display the fps
            cv2.putText(frameDelBg, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            #cv2.imshow("output", frameDelBg)

            # drawimg = mask.copy()
            trakSart = timer()
            #good_new, good_old, drawimg = obj.lkLightflow_track(preFrame, frame, mask)
            good_new, good_old, drawimg = obj.lkLightflow_track(preframeDelBg, frameDelBg, premask)
            #print("good_new.shape:", good_new.shape)
            #print("good_old.shape:", good_old.shape)
            cv2.imshow("res", drawimg)
            # copy the current frame as preFrame for next use
            #preFrame = frame.copy()
            preframeDelBg = frameDelBg.copy()
            preFrame = frame.copy()
            # cv2.waitKey(10)
            ellapseTime = timer() - timeStart
            trackCostTime = timer() - trakSart
            print("trackCostTime:", trackCostTime)
            # update the timeStart and clear tqqhe mask
            if ellapseTime > 3:
                timeStart = timer()
                mask = np.zeros_like(preFrame)  # only detect the motion object
                premask = np.zeros_like(preFrame)

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
"""
a = cv2.imread("E:\\EvolzedArmlogic\\armlogic\\src\\Image\\imageProcess\\1.jpg")
b = cv2.imread("E:\\EvolzedArmlogic\\armlogic\\src\\Image\\imageProcess\\5.jpg")
# c = cv2.imread("E:\\EvolzedArmlogic\\armlogic\\src\\Image\\imageProcess\\5.jpg")


# print(a.type())

# print(b.type())

# wait for test multi bottles track

mask = np.zeros_like(a)
good_new, good_old, drawimg = obj.lkLightflow_track(a, b, mask)
cv2.imshow("a", a)
cv2.imshow("b", b)
cv2.imshow("res", drawimg)
cv2.waitKey()
"""


