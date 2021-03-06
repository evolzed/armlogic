import os
import sys
sys.path.append(os.path.abspath("../../../"))
import gc
import cv2
import numpy as np
import random
from src.BS02.camera import Camera
from timeit import default_timer as timer
from src.BS02.video import Video
from src.BS02.interface import imageCapture
from src.BS02.track import track
from src.BS02.yolo.Yolo import *

# TemplateDir = 'E:\\1\\template.jpg'
TemplateDir = 'template.jpg'
needMakeTemplate = False
feature_params = dict(maxCorners=30,
                      qualityLevel=0.3,
                      minDistance=7,  # min distance between corners
                      blockSize=7)  # winsize of corner
# params for lk track
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


class ImgProc:
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

        self.MAX_CORNERS = 50
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

        # del self.bgVector  #删了 就不能反复使用了
        # gc.collect()

        # cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\h.jpg", self.IhiF)
        # cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\l.jpg", self.IlowF)

    def reStudyBgModelFromCam(self, cam):
        self.studyBackgroundFromCam(cam)
        self.createModelsfromStats()

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
        frame = avi.getImageFromVideo()
        over_flag = 1
        pic_cnt = 0
        frame_cnt = 0
        while frame is not None and over_flag == 1:
            frame = avi.getImageFromVideo()
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
        dst = cv2.GaussianBlur(dst, (19, 19), 3)  #删掉会影响功能
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
                # contourhull = cv2.convexHull(contours[i])  # 凸包
                # cv2.polylines(self.show, [contourhull], True, (500, 255, 0), 2)
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
                # img = cv2.rectangle(self.show, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画矩形
                rows, cols = src.shape[:2]  # shape 0 1 #得出原图的行 列 数
                # [vx, vy, x, y] = cv2.fitLine(contours[i], cv2.DIST_L2, 0, 0.01, 0.01)  # 对轮廓进行多边形拟合
                # lefty = int((x * vy / vx) + y)
                # righty = int(((cols - x) * vy / vx) + y)
                # print("pre final"y
                # rectArray = np.append(rectArray, contourBndBox, axis=0)
                elem = [contourBndBox, contourArea, arclenth, contourCenterGx, contourCenterGy]
                rectArray.append(elem)  # 存轮廓信息到数组中 会对使用它的地方造成影响

                # print("final")
                # print("rectArray", rectArray)
        return rectArray, dst


    def backgroundDiffFast(self, src0):
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

        # tmp = 255 * np.ones(shape=dst.shape, dtype=dst.dtype)

        # inverse the  pixel value to make the mask
        dst = cv2.subtract(255, dst)

        # filter  and morph again and then find the bottle contours
        # dst = cv2.GaussianBlur(dst, (19, 19), 3)  #删掉会影响功能
        # print("is this ok04?")
        dst = cv2.dilate(dst, self.kernel19)
        dst = cv2.dilate(dst, self.kernel19)
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, self.kernel13)  # eclipice
        dst = cv2.erode(dst, self.kernel19)  # eclipice
        # 解决cv2版本3.4.2和4.1.2兼容问题
        # if cv2.__version__.startswith("3"):
        #     _, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # elif cv2.__version__.startswith("4"):
        #     contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(self.show, contours, -1, (0, 255, 0), 3)
        # contourLen = len(contours)
        # print(contourLen)
        momentList = []
        pointList = []

        return None, dst

    def delBg(self, src):
        """
        use the mask pic bgMask to make bit and operation to the cam frame to get a pic that del the bacground

        :param src: input cam pic waited for segment
        :return: rectArray:all boundingboxes of all bottles
                dst:segment result of mask
        """
        # prev_time = timer()
        # simply output the frame that delete the background
        # dst = np.zeros(shape=(960, 1280, 3), dtype=np.uint8)  flexible the dst shape to adapt the src shape
        ##### dst = np.zeros(shape=src.shape, dtype=src.dtype)
        # resarray, bgMask = self.backgroundDiff(src, dst)  # 对src 帧 减去背景 结果放到dst，获得瓶子的框，和掩膜图像
        resarray, bgMask = self.backgroundDiffFast(src)
        # bit and operation
        frame_delimite_bac = cv2.bitwise_and(src, src, mask=bgMask)  # 用掩膜图像和原图像做像素与操作，获得只有瓶子的图
        # curr_time = timer()
        # calculate the cost time
        # exec_time = curr_time - prev_time
        # self.bgTimeCost = exec_time
        # print("Del background Cost time:", self.bgTimeCost)
        return frame_delimite_bac, bgMask, resarray

    def findTrackedOffsetOneBottle(self, good_new, good_old):
        """
        find the offset of one bottle that tracked
        :param good_new:
        good_new is good tracked points of one bottle calculated in the function of trackObj
        :param good_old:
        good_old is good tracked points of one bottle calculated in the function of trackObj
        :return: offset
        offset x and offset y of the tracked bottle
        """
        offsetArray = good_new - good_old
        disTemp = np.sum((good_new - good_old) ** 2, axis=1)
        # print("disTemp", disTemp)
        dis = disTemp.reshape(disTemp.shape[0], 1)
        # print("dis", dis)
        sortIndex = np.argsort(dis, axis=0)
        # print("sortIndex", sortIndex)
        dis_con = np.concatenate((offsetArray, sortIndex), axis=1)
        # print("dis_con", dis_con)
        offsetTemp = dis_con[dis_con[:, 2] == 0]
        # print("offset", offsetTemp)
        offset = []
        offset.append(offsetTemp[0, 0])
        offset.append(offsetTemp[0, 1])
        return offset
        # print(findTrackedOffsets"offset", offset)

    def findTrackedOffsets(self, good_new_con, good_old_con, good_label):
        """
        find all bottle's offset and store in the list
        :param good_new_con:
        :param good_old_con:
        :param good_label:
        :return:targetList
        """
        label_list = np.unique(good_label)
        label_list = label_list[label_list != -1]
        # print("label_list", label_list)
        targetList = []
        for i in label_list:
            # print("i", i)
            # print("good_new_con[:, 2]", good_new_con[:, 2])
            good_new_con_i = good_new_con[abs(good_new_con[:, 2] - i) < 0.0001]  # float modify
            # print("good_new_con", good_new_con)
            good_new = good_new_con_i[:, 0:2]
            # print("good_new", good_new)
            good_old_con_i = good_old_con[abs(good_old_con[:, 2] - i) < 0.0001]
            # print("good_old_con", good_old_con)
            good_old = good_old_con_i[:, 0:2]
            # print("good_old", good_old)
            if np.size(good_new) > 0:
                offset = self.findTrackedOffsetOneBottle(good_new, good_old)
                # print("offset", offset)
                elem = [offset, i]
                # print("elem", elem)
                targetList.append(elem)
        return targetList

    def findTrackedCenterPoint(self, p0, label):
        """
        find  center point of  every bottle's tracked points
        :param p0:
        :param label:
        :return: center_list
        the center point of every bottle store in a list
        """
        if p0 is None:
            return None
        center_list = []
        p0_con = np.concatenate((p0, label), axis=2)
        # print("p0_con_this", p0_con)
        label_list = np.unique(label)
        label_list = label_list[label_list != -1]
        # print("label_list", label_list)
        for i in label_list:
            p0_con_i = p0_con[p0_con[:, :, 2] == i]
            # print("p0_con_i", p0_con_i)
            x = int(np.median(p0_con_i[:, 0]))
            y = int(np.median(p0_con_i[:, 1]))
            indexMask = (p0_con[:, :, 2] == i)  #用于找出Label等于i 的p0的索引
            center_i = [x, y, i, indexMask]
            # print("center_i", center_i)
            center_list.append(center_i)
        return center_list

    def filterBgBox(self, resarray, drawimg):
        detectedBox = []
        for elem in resarray:
            left = elem[0][0]
            top = elem[0][1]
            right = left + elem[0][2]
            bottom = top + elem[0][3]
            contourArea = elem[1]
            # print("area:", contourArea)
            #淘汰掉比较小的
            if contourArea > 5000:
                cv2.rectangle(drawimg, (left, top), (right, bottom), (255, 255, 0))
                newelem = (left, top, right, bottom)
                detectedBox.append(newelem)
        return detectedBox

    def cropBgBoxToYolo(self, detectedBox, src, yolo ,nFrame ,t):
        boxLenth = len(detectedBox)
        objFrameList = []
        if boxLenth > 0:
            for i in range(boxLenth):
                left = detectedBox[i][0]
                top = detectedBox[i][1]
                right = detectedBox[i][2]
                bottom = detectedBox[i][3]
                #图，和偏移坐标 用来恢复
                obj = [src[top:bottom, left:right].copy(), left, top]
                # cv2.imshow(str(i), obj[0])
                #背景切割出来的不同图片
                objFrameList.append(obj)

        objCNNList = []
        kk = 0
        if len(objFrameList) > 0:
            for i in range(len(objFrameList)):
                kk += 1
                imgM = PImage.fromarray(objFrameList[i][0])  # PImage: from PIL import Vision as PImage
                dataDictM = yolo.detectImage(imgM)
                print("small TIME COST:", dataDictM["timeCost"])
                dataDictM["image"] = imgM  # img：Image对象 重复了
                dataDictM["nFrame"] = nFrame
                dataDictM["frameTime"] = t  # 相机当前获取打当前帧nFrame的时间t
                if "box" in dataDictM:
                    print("in kkkkkkkkkk")
                    for k in range(len(dataDictM["box"])):
                        left = dataDictM["box"][k][2]
                        top = dataDictM["box"][k][3]
                        right = dataDictM["box"][k][4]
                        bottom = dataDictM["box"][k][5]
                        modify_x = objFrameList[i][1]
                        modify_y = objFrameList[i][2]

                        left += modify_x
                        top += modify_y
                        right += modify_x
                        bottom += modify_y

                        dataDictM["box"][k][2] = left
                        dataDictM["box"][k][3] = top
                        dataDictM["box"][k][4] = right
                        dataDictM["box"][k][5] = bottom

                    objCNNList.append(dataDictM)
                    # arrM = np.asarray(dataDictM["image"])
                    # cv2.imshow(str(kk), arrM)

        return objCNNList


    def detectObjNotRelyCnn(self, featureimg, drawimg, detectedBox, label_num):
        p0 = cv2.goodFeaturesToTrack(featureimg, mask=None, **feature_params)
        if p0 is not None and len(detectedBox) != 0:
            # trackDict, trackDict = trackObj.createTarget()
            # 画出每个点
            for k in range(p0.shape[0]):
                a = int(p0[k, 0, 0])
                b = int(p0[k, 0, 1])
                cv2.circle(drawimg, (a, b), 3, (0, 255, 255), -1)
            # init the label
            # 构造label的形状
            label = np.ones(shape=(p0.shape[0], 1, 1), dtype=p0.dtype) * (-1)
            # print("len(dataDict[box])", len(dataDict["box"]))

            boxLenth = len(detectedBox)
            # classify  the label  by the dataDict boxes and label them
            if boxLenth > 0:
                for i in range(boxLenth):
                    # print("in!!!!!!!!!!!!!!!!!!!!!!!!!in!!!!!!!!!!!!!!!")
                    left = detectedBox[i][0]
                    top = detectedBox[i][1]
                    right = detectedBox[i][2]
                    bottom = detectedBox[i][3]
                    cv2.rectangle(drawimg, (left, top), (right, bottom), (255, 255, 0))
                    # store every point label
                    # print("iiiiiiiiiiiiiiiiiiiiiiiiii------------:", i)
                    for k in range(p0.shape[0]):
                        # print("p0", p0[k, 0, 0])
                        # print("p1", p0[k, 0, 1])
                        if (left - 20 <= p0[k, 0, 0]) and \
                                (p0[k, 0, 0] <= right + 20) and \
                                (top - 20 <= p0[k, 0, 1]) and \
                                (p0[k, 0, 1] <= bottom + 20):
                            label[k, 0, 0] = i
                print("label", label)
                print("unique", np.unique(label[label != -1]))
                # num is the detected label number
                if (label != -1).any() and np.size(np.unique(label[label != -1])) >= label_num:
                    # flag = 1
                    centerL = self.findTrackedCenterPoint(p0, label)
                    allList = []
                    for x in centerL:
                        allList.append([x[0], x[1], x[2], 0, 0])

                    return p0, label, allList
                else:
                    return None, None, None
            else:
                return None, None, None
        else:
            return None, None, None

    # boxList.append([predicted_class, score, left, top, right, bottom, \
    #                 angle, diameter, centerX, centerY, trackID, deltaX, deltaY, speedX, speedY])

    def detectObjNotRelyLKFromDict(self, featureimg, drawimg, dataDict):
        """
         detect the points and  add the labels on every point,and then track them,the label_num define the min count of detected boxes

        :param featureimg: feature image, pre image, the  points of this  image(p0) will be track in the trackObj cycle if the points is labeled
        :param drawimg: drawing img,which is used for draw
        :param dataDict: the dataDict retruned by image check
        :param feature_params:track params
        :param label_num: the min detected boxes
        :return:p0, label, allList
        p0 is detected and labeled points  and will be track in the trackObj cycle ,
        label is the label of p0 points
        allList is a list to store center point position and offset and id
        allList[seqN][0], allList[seqN][1] is center point position
        allList[seqN][3], allList[seqN][4] is offset X offset Y
        allList[seqN][2], is id
        """
        #detect the points
        # trackObj = Track()
        allList = []
        count = 0
        if "box" in dataDict:
            boxLenth = len(dataDict["box"])
            # classify  the label  by the dataDict boxes and label them
            if boxLenth > 0:
                for i in range(len(dataDict["box"])):
                    # print("in11111111111")
                    # if "box" in dataDict and dataDict["box"][i][1] > 0.9 and dataDict["box"][i][3] > 180:
                    if "box" in dataDict and dataDict["box"][i][1] > 0.9:
                        # print("in2222222222")
                        left = dataDict["box"][i][2]
                        top = dataDict["box"][i][3]
                        right = dataDict["box"][i][4]
                        bottom = dataDict["box"][i][5]
                        cv2.rectangle(drawimg, (left, top), (right, bottom), (255, 255, 0))
                        # store every point label
                        # print("iiiiiiiiiiiiiiiiiiiiiiiiii------------:", i)
                        centerX = int((left + right)/2)
                        centerY = int((top + bottom) / 2)

                        allList.append([centerX, centerY, int(count), 0, 0])

                        dataDict["box"][i][8] = centerX
                        dataDict["box"][i][9] = centerY
                        dataDict["box"][i][10] = int(count)

                        print("count", count)
                        count += 1
                return allList
            else:
                return None
            return allList
        else:
            return None

    def detectObjNotRelyLK(self, featureimg, drawimg, dataDictList):
        """
         detect the points and  add the labels on every point,and then track them,the label_num define the min count of detected boxes

        :param featureimg: feature image, pre image, the  points of this  image(p0) will be track in the trackObj cycle if the points is labeled
        :param drawimg: drawing img,which is used for draw
        :param dataDict: the dataDict retruned by image check
        :param feature_params:track params
        :param label_num: the min detected boxes
        :return:p0, label, allList
        p0 is detected and labeled points  and will be track in the trackObj cycle ,
        label is the label of p0 points
        allList is a list to store center point position and offset and id
        allList[seqN][0], allList[seqN][1] is center point position
        allList[seqN][3], allList[seqN][4] is offset X offset Y
        allList[seqN][2], is id
        """
        #detect the points
        # trackObj = Track()
        allList = []
        count = 0
        for index in range(len(dataDictList)):
            dataDict = dataDictList[index]
            if "box" in dataDict:
                boxLenth = len(dataDict["box"])
                # classify  the label  by the dataDict boxes and label them
                if boxLenth > 0:
                    for i in range(len(dataDict["box"])):
                        # print("in11111111111")
                        # if "box" in dataDict and dataDict["box"][i][1] > 0.9 and dataDict["box"][i][3] > 180:
                        if "box" in dataDict:
                            # print("in2222222222")
                            left = dataDict["box"][i][2]
                            top = dataDict["box"][i][3]
                            right = dataDict["box"][i][4]
                            bottom = dataDict["box"][i][5]
                            cv2.rectangle(drawimg, (left, top), (right, bottom), (255, 255, 0))
                            # store every point label
                            print("iiiiiiiiiiiiiiiiiiiiiiiiii------------:", i)
                            centerX = int((left + right)/2)
                            centerY = int((top + bottom) / 2)

                            allList.append([centerX, centerY, int(count), 0, 0])

                            dataDict["box"][i][8] = centerX
                            dataDict["box"][i][9] = centerY
                            dataDict["box"][i][10] = int(count)
                            print("count", count)
                            count += 1
        return allList



    def detectObj(self, featureimg, drawimg, dataDict):
        """
         detect the points and  add the labels on every point,and then track them,the label_num define the min count of detected boxes

        :param featureimg: feature image, pre image, the  points of this  image(p0) will be track in the trackObj cycle if the points is labeled
        :param drawimg: drawing img,which is used for draw
        :param dataDict: the dataDict retruned by image check
        :param feature_params:track params
        :param label_num: the min detected boxes
        :return:p0, label, allList
        p0 is detected and labeled points  and will be track in the trackObj cycle ,
        label is the label of p0 points
        allList is a list to store center point position and offset and id
        allList[seqN][0], allList[seqN][1] is center point position
        allList[seqN][3], allList[seqN][4] is offset X offset Y
        allList[seqN][2], is id
        """
        #detect the points
        # trackObj = Track()
        theList =[]
        p0 = cv2.goodFeaturesToTrack(featureimg, mask=None, **feature_params)
        if p0 is not None and "box" in dataDict:
            # trackDict, trackDict = trackObj.createTarget()
            #画出每个点
            for k in range(p0.shape[0]):
                a = int(p0[k, 0, 0])
                b = int(p0[k, 0, 1])
                cv2.circle(drawimg, (a, b), 3, (0, 255, 255), -1)
            # init the label
            #构造label的形状
            label = np.ones(shape=(p0.shape[0], 1, 1), dtype=p0.dtype) * (-1)
            # print("len(dataDict[box])", len(dataDict["box"]))

            boxLenth = len(dataDict["box"])
            # classify  the label  by the dataDict boxes and label them
            if boxLenth > 0:
                for i in range(len(dataDict["box"])):
                    if "box" in dataDict and dataDict["box"][i][1] > 0.92 and dataDict["box"][i][3] > 180:
                        # print("in!!!!!!!!!!!!!!!!!!!!!!!!!in!!!!!!!!!!!!!!!")
                        left = dataDict["box"][i][2]
                        top = dataDict["box"][i][3]
                        right = dataDict["box"][i][4]
                        bottom = dataDict["box"][i][5]

                        centerX = int((left + right) / 2)
                        centerY = int((top + bottom) / 2)

                        dataDict["box"][i][8] = centerX
                        dataDict["box"][i][9] = centerY
                        dataDict["box"][i][10] = int(i)

                        theList.append([centerX, centerY, int(i), 0, 0])


                        # cv2.rectangle(drawimg, (left, top), (right, bottom), (255, 255, 0))
                        # store every point label
                        # print("iiiiiiiiiiiiiiiiiiiiiiiiii------------:", i)
                        for k in range(p0.shape[0]):
                            # print("p0", p0[k, 0, 0])
                            # print("p1", p0[k, 0, 1])
                            if (left - 20 <= p0[k, 0, 0]) and \
                                    (p0[k, 0, 0] <= right + 20) and \
                                    (top - 20 <= p0[k, 0, 1]) and \
                                    (p0[k, 0, 1] <= bottom + 20):
                                label[k, 0, 0] = int(i)
                # print("label", label)
                # print("unique", np.unique(label[label != -1]))
                # num is the detected label number
                # 或操作 只要有一个-1 就是 True   不是-1的大于0
                if (label != -1).any() and np.size(np.unique(label[label != -1])) >= 0:
                # if (label != -1).any():
                    # flag = 1
                    centerL = self.findTrackedCenterPoint(p0, label)
                    allList = []
                    for x in centerL:
                        allList.append([x[0], x[1], x[2], x[3], 0, 0])# is not used  temparary
                        indexLabel = int(x[2])
                        # print("indexLabel:", indexLabel)
                        # dataDict["box"][indexLabel][8] = x[0]      #centerX
                        # dataDict["box"][indexLabel][9] = x[1]       #centerY
                        # dataDict["box"][indexLabel][10] = int(x[2])   #trackID

                    #return 的时机不对
                else:
                    return None, None, None, None, dataDict

                return p0, label, allList, theList, dataDict
            else:
                return None, None, None,None, dataDict
        else:
            return None, None,None, None, dataDict


    def trackObj(self, featureimg, secondimg, drawimg, label, p0, deltaT):
        """
        track the obj of deteced, input the deteced points or the last tracked points,output the new tracked points and its labels

        :param featureimg: feature image, pre image,the  points of this  image(p0) will be track in the second image(p1)
        :param secondimg:  second image,current image, will become pre image in the next cycle
        the tracked points of this image(p1) will become p0 again in the next cycle
        :param drawimg:  drawing img,which is used for draw
        :param label: label of points by detectObj
        :param p0:  the target points for track by detectObj
        :param lk_params:  track params
        :return: p0, label: p0 is tracked points and will be track for next cycle,label is the label of p0 points
                allList is a list to store center point position and offset and id
                allList[seqN][0], allList[seqN][1] is center point position
                allList[seqN][3], allList[seqN][4] is offset X offset Y
                allList[seqN][2], is id
        """
        #num of track
        if p0 is not None and np.size(p0.shape[0]) > 0:
            # track the pre image points p0 to get the tracked points of p1
            p1, st, err = cv2.calcOpticalFlowPyrLK(featureimg, secondimg, p0, None, **lk_params)
            if p1 is not None and (label != -1).any() and np.size(p1.shape[0]) > 0:
                # print("st", st)
                #find the good tracked points
                good_new = p1[st == 1]  # will error when twise  can not use the same
                good_old = p0[st == 1]  # will error when twise

                good_label = label[st == 1]
                # print("good_label", good_label)
                #concatenate the points and their labels not used
                good_new_con = np.concatenate((good_new, good_label), axis=1)
                good_old_con = np.concatenate((good_old, good_label), axis=1)
                # print("good_new_con", good_new_con)
                if good_new_con is not None:
                    targetlist = self.findTrackedOffsets(good_new_con, good_old_con, good_label)
                    # print("targetlist", targetlist)

                #unfold the points and draw it
                for i, (new, old) in enumerate(zip(good_new_con, good_old_con)):  # fold and enumerate with i
                    a, b, la = new.ravel()  # unfold
                    c, d, la = old.ravel()
                    a = int(a)
                    b = int(b)
                    c = int(c)
                    d = int(d)
                    if la != -1:
                        cv2.line(drawimg, (a, b), (c, d), (0, 255, 255), 1)

                p0 = good_new.reshape(-1, 1, 2)
                label = good_label.reshape(-1, 1, 1)

                centerList = self.findTrackedCenterPoint(p0, label)

                # print("centerList", centerList)

                allList = []
                if centerList is not None:
                    for center in centerList:
                        for i in range(len(targetlist)):   #为了加上
                            if center[2] == targetlist[i][1]:  #CNN label 相等
                                # 坐标 label 速度
                                allList.append([center[0], center[1], center[2], center[3], #索引
                                                round(targetlist[i][0][0], 3), round(targetlist[i][0][1], 3),#offset
                                                round(targetlist[i][0][0]/deltaT, 3),  #speed
                                                round(targetlist[i][0][1]/deltaT, 3),
                                                ]) #跟踪起始时间 如果和dataDict[frameTime]相等，
                                # 则代表跟踪那一次dataDict的数据
                        # print("center", center)
                return p0, label, allList, True
            else:
                return None, None, None, False
        else:
            return None, None, None, False



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
        w = self.eDistance(rbox[0], rbox[1])
        h = self.eDistance(rbox[1], rbox[2])

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


    # analyse every point
    def analyseTrackPoint(self, good_new, good_old, precisionThreshold):
        # offset = np.array([0, 0])
        # return good_new, good_old, offset
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
        if np.isnan(good_new).sum() > 0 or np.isnan(good_old).sum() > 0:
            return good_new, good_old, np.array([0, 0])
        good_new0 = np.array([[0, 0]])
        good_old0 = np.array([[0, 0]])
        pointLen = good_new.shape[0]
        if pointLen == 0:
            return good_new, good_old, np.array([0, 0])
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
        print(np.array([good_new[0]]))
        for i in index:
            good_new0 = np.append(good_new0, np.array([good_new[i]]), axis=0)
            good_old0 = np.append(good_old0, np.array([good_old[i]]), axis=0)
        good_new0 = np.delete(good_new0, 0, axis=0)
        good_old0 = np.delete(good_old0, 0, axis=0)
        good_new0 = good_new0.astype(int)
        good_old0 = good_old0.astype(int)

#calcute the ave speed
        pointLen0 = good_new0.shape[0]
        if pointLen0 == 0:
            return good_new, good_old, np.array([0, 0])
        #speedarray = np.array([0, 0])
        # for i in range(pointLen):
        #     speed = good_new0[i] - good_old0[i]
        #     speedarray = np.append(speedarray, speed)
        # speedarray = np.delete(speedarray,0,axis=0)
        speedarray = good_new0 - good_old0
        pointLen0 = good_new0.shape[0]
        print("offset_diff_array", speedarray)
        print("offset_array_sum", np.sum(speedarray, axis=0))
        speed = np.sum(speedarray, axis=0)/pointLen0

        #print()
        return good_new0, good_old0, speed


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

    def lkLightflow_track(self, featureimg, secondimg_orig, mask,inputCorner):
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
        # change to gray
        featureimg = cv2.cvtColor(featureimg, cv2.COLOR_BGR2GRAY)
        secondimg = cv2.cvtColor(secondimg_orig, cv2.COLOR_BGR2GRAY)
        corner_count = self.MAX_CORNERS
        # find the good corners for track
        if inputCorner is None:
            cornersA = cv2.goodFeaturesToTrack(featureimg, mask=None, **feature_params)
            if cornersA is None:
                return None, None, 0, secondimg_orig
        else:
            cornersA = inputCorner.copy()

        cornersB, st, err = cv2.calcOpticalFlowPyrLK(featureimg, secondimg, cornersA, None, **lk_params)
        # find the point that concerned to be tracked
        offset = np.array([0, 0])
        drawimg = secondimg_orig.copy()
        if cornersB is None:
            return None, None, offset, drawimg
        good_new = cornersB[st == 1]
        good_old = cornersA[st == 1]

        img = np.zeros_like(mask)

        # draw line between the tracked corners of qpre frame and current frameq
        for i, (new, old) in enumerate(zip(good_new, good_old)):  # fold and enumerate with i
            a, b = new.ravel()  # unfold
            c, d = old.ravel()
            # mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 1)
            if mask is not None:
                mask = cv2.line(mask, (a, b), (c, d), (0, 0, 255), 1)

            if self.eDistance(np.array(list((a, b))), np.array(list((c, d)))) > 1:
                # drawimg = cv2.circle(drawimg, (a, b), 5, color[i].tolist(), -1)
                drawimg = cv2.circle(drawimg, (a, b), 5, (0, 0, 255), -1)   #red
            if mask is not None:
                img = cv2.add(drawimg, mask)
                print("test:", a-int(offset[0]))
                print("type:", type(int(a-offset[0])))

            else:
                cv2.line(drawimg, (a, b), (c, d), (0, 0, 255), 1)
                if offset is not None:
                    cv2.line(drawimg, (a, b), (int(a - offset[0]), int(b - offset[1])), (0, 255, 255), 1)  # yellow
        img = drawimg

        return good_new, good_old, offset, img

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
    obj = ImgProc(50, imgCapObj)
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

