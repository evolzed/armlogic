import os
import sys
sys.path.append(os.path.abspath("../../../"))
import gc
import cv2
import numpy as np
import random
from src.Vision.camera import Camera
from timeit import default_timer as timer
from src.Vision.video import Video
from src.Vision.interface import imageCapture
from src.Track import Track
from src.Vision.imageProcess.imageTools import *
# TemplateDir = 'E:\\1\\template.jpg'
TemplateDir = 'template.jpg'
needMakeTemplate = False



class BgLearn:
    def __init__(self, bgStudyNum, imgCapObj):
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


        self.win_size = 10
        self.imgCap = imgCapObj


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

    def createModelsfromStats(self, scale=9.0):
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

    def studyBackground(self):
        frame = self.imgCap.getBgImage()
        over_flag = 1
        pic_cnt = 0
        frame_cnt = 0
        while frame is not None and over_flag == 1:
            frame, nFrame, t = self.imgCap.getBgImage()
            if frame_cnt % 20 == 0:  # get a frame per 20
                fin = np.float32(frame)
                self.bgVector[pic_cnt] = fin
                pic_cnt += 1
            frame_cnt += 1
            print("pic_cnt", pic_cnt)
            if pic_cnt == self.BG_STUDY_NUM:
                over_flag = 0
        for i in range(self.bgVector.shape[0]):
            self.avgBackground(self.bgVector[i])



    def studyBackgroundFromVideo(self, videoDir):
        """
        get many pics for time interval of 60sec by cam and store the pics in  bgVector.
        then  call the avgBackground method

        :param cam: input camera object
        :return: None
        """
        avi = Video(videoDir)
        frame = avi.getImageFromVedio()
        over_flag = 1
        pic_cnt = 0
        frame_cnt = 0
        while frame is not None and over_flag == 1:
            frame = avi.getImageFromVedio()
            if frame_cnt % 20 == 0:  # get a frame per 20
                fin = np.float32(frame)
                self.bgVector[pic_cnt] = fin
                pic_cnt += 1
            frame_cnt += 1
            # print("shape", fin.shape)
            # store the frame in list bgVector


            # print("pic_cnt", pic_cnt)
            # wait about 200 milli seconds
            # cv2.waitKey(20)
            print("pic_cnt", pic_cnt)
            if pic_cnt == self.BG_STUDY_NUM:
                over_flag = 0
        for i in range(self.bgVector.shape[0]):
            # print("i", i)
            self.avgBackground(self.bgVector[i])

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
                if pic_cnt == self.BG_STUDY_NUM:
                    over_flag = 0

            # print("shapebg", self.bgVector.shape)
            for i in range(self.bgVector.shape[0]):
                # print("i", i)
                self.avgBackground(self.bgVector[i])
        except Exception as e:
            print(e)
            # when occur exception ,the camera will disconnect
            cam.destroy()

    def filterBgBox(self, resarray, drawimg):
        detectedBox = []
        for elem in resarray:
            left = elem[0][0]
            top = elem[0][1]
            right = left + elem[0][2]
            bottom = top + elem[0][3]
            contourArea = elem[1]
            print("area:", contourArea)
            # 淘汰掉比较小的
            if contourArea > 5000:
                cv2.rectangle(drawimg, (left, top), (right, bottom), (255, 255, 0))
                newelem = (left, top, right, bottom)
                detectedBox.append(newelem)
        return detectedBox


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

        # segment the src through IlowF and IhiF
        dst = cv2.inRange(src, self.IlowF, self.IhiF)
        # cv2.imshow("segment_debug", dst)

        # morph process the frame to clear the noise and highlight our object region

        dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, self.kernel7)

        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, self.kernel7)

        tmp = 255 * np.ones(shape=dst.shape, dtype=dst.dtype)

        # inverse the  pixel value to make the mask
        dst = cv2.subtract(255, dst)

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
                arclenth = cv2.arcLength(contours[i], True)  # 周长

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
                # print("pre final"y
                # rectArray = np.append(rectArray, contourBndBox, axis=0)
                elem = [contourBndBox, contourArea, arclenth, contourCenterGx, contourCenterGy]
                rectArray.append(elem)  # 存轮廓信息到数组中 会对使用它的地方造成影响

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

    def correctAngle(self, rbox):
        """
        correct the angle to -90 to 90 for get Pose,

        :param rbox: input rotatebox
        :return: angle: angle that modified
        """
        w = eDistance(rbox[0], rbox[1])
        h = eDistance(rbox[1], rbox[2])

        # 钝角 内积小于0
        xAxisVector = np.array([[0], [1]])
        angle = 0
        # find the long side of rotate rect
        if w > h:
            # find the low point the vector is from low to high
            v = np.zeros(rbox[0].shape, dtype=rbox.dtype)

            v = rbox[1] - rbox[0]
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
            # 内积大于0 是锐角 小于0是钝角
            if np.dot(v, xAxisVector)[0] > 0:
                angle = - angle
            if np.dot(v, xAxisVector)[0] < 0:
                angle = - angle + 90
            if angle == 0:
                angle = 90

        if 0 <= angle <= 90:
            angle = angle
        if 0 < angle < 180:
            angle = angle - 180
        return angle

    def getBoxOnlyPic(self, dataDict, frameOrg0):
        """
        get the pictures of only have one yolo detected box,return the list of these pictures
        :param dataDict:
        :param frameOrg0:
        :return:
        """
        frameOrg = frameOrg0.copy()
        list = []
        if "box" in dataDict:
            for i in range(len(dataDict["box"])):
                if dataDict["box"][i][1] > 0.8:
                    # get the box vertex
                    left = dataDict["box"][i][2]
                    top = dataDict["box"][i][3]
                    right = dataDict["box"][i][4]
                    bottom = dataDict["box"][i][5]
                    rectTop = np.array([left, top])
                    rectBottle = (right - left, bottom - top)
                    # get the BOX ROI from frame
                    x = int((left + right)/2)
                    y = int((top + bottom)/2)
                    w = abs(left - right)
                    h = abs(bottom - top)
                    roi = np.zeros(4, dtype=int)
                    roi[0] = x
                    roi[1] = y
                    roi[2] = w
                    roi[3] = h
                    maskroi = creatMaskFromROI(frameOrg, roi)
                    show = cv2.bitwise_and(frameOrg, frameOrg, mask=maskroi)
                    list.append(show)
        if len(list) == 0:
            list.append(frameOrg)
        return list


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
                    contourLen = len(contours)
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

                        rbox = rbox + rectTop
                        rbox = np.int0(rbox)
                        # 画出来
                        cv2.drawContours(frameOrg, [rbox], 0, (255, 0, 255), 1)
                        # cv2.imshow("pos", frameOrg)
                        # store angle and diameter to the dataDict
                        dataDict["box"][i][6] = angle
                        dataDict["box"][i][7] = diameter
        curr_time = timer()
        exec_time = curr_time - prev_time  # 计算图像识别的执行时间
        dataDict["getPosTimeCost"] = exec_time
        return dataDict

    def findContourMatch(self, frameDelBg):
        cx0 = -1
        cy0 = -1
        contour=None
        index =-1
        # src_show = frame.copy()
        src_copy = frameDelBg.copy()  # float change to np int
        src_copy_gray = cv2.cvtColor(src_copy, cv2.COLOR_BGR2GRAY)

        # threshold1 = cv2.getTrackbarPos('threshold1', 'Canny')
        # threshold2 = cv2.getTrackbarPos('threshold2', 'Canny')

        # edge = cv2.Canny(src_copy_gray, threshold1, threshold2)
        edge = cv2.Canny(src_copy_gray, 78, 148)

        if cv2.__version__.startswith("3"):
            _, contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        elif cv2.__version__.startswith("4"):
            contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # pic = src_show.copy()
        # print("hierarchy", hierarchy)
        # hierarchy_choose = hierarchy[hierarchy[:, 1, 0] == -1]
        # 3 parent contour
        # matchRetList = []
        if len(contours) > 0:
            for ci in range(len(contours)):
                # if contourTemplate is not None:
                matchRet = cv2.matchShapes(contours[ci], contourTemplate, 1, 0.0)
                # matchRetList.append(matchRet)
                print("matchRet", matchRet)
                # matchRetListD = np.array(matchRetList)
                # print("matchRetList.min", matchRetListD.min())
                # print("matchRetList.min", matchRetListD.argmin())
                # minIndex = matchRetListD.argmin()
                # minMatch = matchRetListD.min()
                arclenth = cv2.arcLength(contours[ci], True)  # 面积
                area = cv2.contourArea(contours[ci])  # 4386.5
                if matchRet < 0.1 and arclenth > 300 and 8000 > area > 5000 and hierarchy[0, ci, 3] != -1:
                    # cv2.drawContours(pic, contours, ci, (0, 255, 0), 6)
                    M = cv2.moments(contours[ci])  # 计算第一条轮廓的各阶矩,字典形式
                    # print (M)
                    # 这两行是计算中心点坐标
                    cx0 = int(M['m10'] / M['m00'])
                    cy0 = int(M['m01'] / M['m00'])
                    # cv2.putText(pic, text=str("milk"), org=(cx, cy),
                    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    #             fontScale=1, color=(0, 0, 255),
                    #             thickness=2)
                    index = ci
                    contour =  contours[ci]
                    break
        return cx0, cy0, contour

    def loadContourTemplate(self, ContourDir):
        contoursGet = np.array([])
        contours = np.array([])
        template = cv2.imread(ContourDir, 0)
        pic = template.copy()
        templateedge = template
        if cv2.__version__.startswith("3"):
            _, contours, hierarchy = cv2.findContours(templateedge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        elif cv2.__version__.startswith("4"):
            contours, hierarchy = cv2.findContours(templateedge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            print("in_______________")
            print("len(contours)", len(contours))
            for ci in range(len(contours)):
                if hierarchy[0, ci, 3] != -1:  # find the most big hierarchy
                    arclenth = cv2.arcLength(contours[ci], True)  # 面积
                    area = cv2.contourArea(contours[ci])  # 4386.5
                    if arclenth > 900 and 10000 > area > 5000:
                        cv2.drawContours(pic, contours, ci, (255, 255, 255), 6)
                        contoursGet = contours[ci]
                        print("arcle", arclenth)
                        print("area", area)
        cv2.imshow("tem", pic)
        return contoursGet

    def makeTemplate(self, frameDelBg, frame, writeDir = None):
        src_show = frame.copy()
        src_copy = frameDelBg.copy()  #float change to np int
        # print("src_copy shape", src_copy.shape)
        # print("src_copy dtype", src_copy.dtype)
        src_copy_gray = cv2.cvtColor(src_copy, cv2.COLOR_BGR2GRAY)
        # print("src_copy_gray shape", src_copy_gray.shape)
        # print("src_copy_gray dtype", src_copy_gray.dtype)
        # ret, binary = cv2.threshold(src_copy_gray, 127, 255, cv2.THRESH_BINARY)
        # threshold1 = cv2.getTrackbarPos('threshold1', 'Canny')
        # threshold2 = cv2.getTrackbarPos('threshold2', 'Canny')
        # edge = cv2.Canny(src_copy_gray, threshold1, threshold2)
        edge = cv2.Canny(src_copy_gray, 78, 148)
        if cv2.__version__.startswith("3"):
            _, contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        elif cv2.__version__.startswith("4"):
            contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        pic = src_show.copy()
        if len(contours) > 0:
            for ci in range(len(contours)):
                if hierarchy[0, ci, 3] != -1:  # find not the most big hierarchy
                    r = random.randint(0, 255)
                    g = random.randint(0, 255)
                    b = random.randint(0, 255)
                    # child = hierarchy[0, ci, 2]
                    arclenth = cv2.arcLength(contours[ci], True)  # 面积
                    area = cv2.contourArea(contours[ci])  # 4386.5
                    # print("arcle", arclenth)
                    # print("area", area)
                    if arclenth > 300 and 8000 > area > 5000:  #the white bottle
                        if writeDir is not None:
                            cv2.imwrite(writeDir, edge)
                        cv2.drawContours(pic, contours, ci, (b, g, r), 3)
                        # cv2.drawContours(pic, contours, ci, (0, 0, 0), 3)
                        print("arcle", arclenth)
                        print("area", area)
        cv2.imshow("pic", pic)

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

def nothing(x):
    pass


if __name__ == "__main__":


    # cam = Camera()
    videoDir = "E:\\1\\3.avi"
    bgDir = "E:\\1\\背景.avi"
    avi = Video(videoDir)
    bgAvi = Video(bgDir)
    imgCapObj = imageCapture(None, avi, bgAvi)
    obj = BgLearn(50, imgCapObj)
    obj.studyBackground()
    # obj.studyBackgroundFromCam(cam)
    # obj.studyBackgroundFromVideo("E:\\1\\背景.avi")
    obj.createModelsfromStats(8.0)

    try:
        # cam = Camera()
        # preFrame, nFrameNum, t = cam.getImage()
        preFrame, nFrameNum, t = obj.imgCap.getImage()
        preframeDelBg, bgmask, resarray= obj.delBg(preFrame)

        #preFrame = np.zeros_like(frame)
        mask = preFrame.copy()

        premask = preframeDelBg.copy()
        timeStart = timer()
        timeCnt = 0
        flag = 1
        feature_params = dict(maxCorners=30,
                              qualityLevel=0.3,
                              minDistance=7,  # min distance between corners
                              blockSize=7)  # winsize of corner
        # params for lk track
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        good_old = np.array([])
        cv2.namedWindow('Canny')
        cv2.createTrackbar('threshold1', 'Canny', 50, 400, nothing)
        cv2.createTrackbar('threshold2', 'Canny', 100, 400, nothing)

        contourTemplate = obj.loadContourTemplate(TemplateDir)
        while 1:
            # frame, nFrameNum, t = cam.getImage()
            frame, nFrameNum, t = obj.imgCap.getImage()
            if frame is None:
                break
            # camra fault tolerant
            # mem addr use is
            # if frame is None:
            #     continue

            obj.show = frame.copy()
            pic = frame.copy()
            # get fps of cam output

            # fps = cam.getCamFps(nFrameNum)

            # use the background model to del the bacground of  framezzzz

            frameDelBg, bgmask, resarray = obj.delBg(frame)
            drawimg = frameDelBg.copy()
            obj.filterBgBox(resarray, drawimg)
            cv2.imshow("today", drawimg)
            if needMakeTemplate:
                obj.makeTemplate(frameDelBg, frame)

            cx, cy, contour = obj.findContourMatch(frameDelBg)
            if cx != -1 and cy != -1 and contour is not None:
                cv2.drawContours(pic, contour, -1, (0, 255, 0), 6)  #can
                cv2.putText(pic, text=str("milk"), org=(cx, cy),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(0, 0, 255),
                            thickness=2)

            cv2.imshow("pic", pic)

            cv2.imshow("show", obj.show)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                pass
            #     cam.destroy()
                break
    except Exception as e:
        print(e)
        pass
        # cam.destroy()

