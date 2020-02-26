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
from src.Track import track
from src.BS02.imageProcess.imageTools import *

# TemplateDir = 'E:\\1\\template.jpg'
TemplateDir = 'template.jpg'
needMakeTemplate = False


class ImgTrack:
    def __init__(self):
        self.feature_params = dict(maxCorners=30,
                              qualityLevel=0.3,
                              minDistance=7,  # min distance between corners
                              blockSize=7)  # winsize of corner
        # params for lk track
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.MAX_CORNERS = 50


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
        print("label_list", label_list)
        targetList = []
        for i in label_list:
            print("i", i)
            print("good_new_con[:, 2]", good_new_con[:, 2])
            good_new_con_i = good_new_con[abs(good_new_con[:, 2] - i) < 0.0001]  # float modify
            # print("good_new_con", good_new_con)
            good_new = good_new_con_i[:, 0:2]
            print("good_new", good_new)
            good_old_con_i = good_old_con[abs(good_old_con[:, 2] - i) < 0.0001]
            # print("good_old_con", good_old_con)
            good_old = good_old_con_i[:, 0:2]
            print("good_old", good_old)
            if np.size(good_new) > 0:
                offset = self.findTrackedOffsetOneBottle(good_new, good_old)
                print("offset", offset)
                elem = [offset, i]
                print("elem", elem)
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
            center_i = [x, y, i]
            # print("center_i", center_i)
            center_list.append(center_i)
        return center_list


    def detectObjNotRelyCnn(self, featureimg, drawimg, detectedBox, label_num):
        p0 = cv2.goodFeaturesToTrack(featureimg, mask=None, **self.feature_params)
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
                    print("in!!!!!!!!!!!!!!!!!!!!!!!!!in!!!!!!!!!!!!!!!")
                    left = detectedBox[i][0]
                    top = detectedBox[i][1]
                    right = detectedBox[i][2]
                    bottom = detectedBox[i][3]
                    cv2.rectangle(drawimg, (left, top), (right, bottom), (255, 255, 0))
                    # store every point label
                    print("iiiiiiiiiiiiiiiiiiiiiiiiii------------:", i)
                    for k in range(p0.shape[0]):
                        print("p0", p0[k, 0, 0])
                        print("p1", p0[k, 0, 1])
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




    def detectObj(self, featureimg, drawimg, dataDict, label_num):
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
        # detect the points
        # trackObj = Track()
        p0 = cv2.goodFeaturesToTrack(featureimg, mask=None, **self.feature_params)
        if p0 is not None and "box" in dataDict:
            # trackDict, trackDict = trackObj.createTarget()
            # 画出每个点
            for k in range(p0.shape[0]):
                a = int(p0[k, 0, 0])
                b = int(p0[k, 0, 1])
                cv2.circle(drawimg, (a, b), 3, (0, 255, 255), -1)
            # init the label
            # 构造label的形状
            label = np.ones(shape=(p0.shape[0], 1, 1), dtype=p0.dtype) * (-1)
            print("len(dataDict[box])", len(dataDict["box"]))

            boxLenth = len(dataDict["box"])
            # classify  the label  by the dataDict boxes and label them
            if boxLenth > 0:
                for i in range(len(dataDict["box"])):
                    if "box" in dataDict and dataDict["box"][i][1] > 0.9 and dataDict["box"][i][3] > 180:
                        print("in!!!!!!!!!!!!!!!!!!!!!!!!!in!!!!!!!!!!!!!!!")
                        left = dataDict["box"][i][2]
                        top = dataDict["box"][i][3]
                        right = dataDict["box"][i][4]
                        bottom = dataDict["box"][i][5]
                        cv2.rectangle(drawimg, (left, top), (right, bottom), (255, 255, 0))
                        # store every point label
                        print("iiiiiiiiiiiiiiiiiiiiiiiiii------------:", i)
                        for k in range(p0.shape[0]):
                            print("p0", p0[k, 0, 0])
                            print("p1", p0[k, 0, 1])
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

    def trackObj(self, featureimg, secondimg, drawimg, label, p0):
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
            p1, st, err = cv2.calcOpticalFlowPyrLK(featureimg, secondimg, p0, None, **self.lk_params)
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
                print("good_new_con", good_new_con)
                if good_new_con is not None:
                    targetlist = self.findTrackedOffsets(good_new_con, good_old_con, good_label)
                    print("targetlist", targetlist)

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

                print("centerList", centerList)

                allList = []
                if centerList is not None:
                    for center in centerList:
                        for i in range(len(targetlist)):
                            if center[2] == targetlist[i][1]:
                                # 坐标 label 速度
                                allList.append([center[0], center[1], center[2], targetlist[i][0][0], targetlist[i][0][1]])
                        print("center", center)
                return p0, label, allList
            else:
                return None, None, None
        else:
            return None, None, None




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
        # generate array of random colors
        # change to gray
        featureimg = cv2.cvtColor(featureimg, cv2.COLOR_BGR2GRAY)
        secondimg = cv2.cvtColor(secondimg_orig, cv2.COLOR_BGR2GRAY)
        corner_count = self.MAX_CORNERS
        # find the good corners for track
        if inputCorner is None:
            cornersA = cv2.goodFeaturesToTrack(featureimg, mask=None, **self.feature_params)
            if cornersA is None:
                return None, None, 0, secondimg_orig
        else:
            cornersA = inputCorner.copy()

        cornersB, st, err = cv2.calcOpticalFlowPyrLK(featureimg, secondimg, cornersA, None, **self.lk_params)
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