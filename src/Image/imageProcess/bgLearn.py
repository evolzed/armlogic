import cv2
import numpy as np
from src.Image.camera import Camera
from timeit import default_timer as timer


# bgLearn
# Description:
# Learn the backgroud by pics from cam then get a background model
# bgLearn is implemented by sequential procedure, and  theses procedure are  expressed  several functions as below.


class Bglearn():
#background learn method class#

    def __init__(self):
        #private attribute of class
        self.BG_STUDY_NUM = 50
        self.bgVector = np.zeros(shape=(self.BG_STUDY_NUM, 960, 1280, 3), dtype=np.float32)
        self.IavgF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)
        # average of frames
        self.IprevF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)
        self.Iscratch2 = np.zeros(shape=(960, 1280, 3), dtype=np.float32)
        self.IdiffF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)
        # average difference of frames
        self.IhiF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)
        self.IlowF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)
        self.Icount = 0
        self.kernel5 = np.ones((5, 5), np.uint8)
        self.kernel7 = np.ones((7, 7), np.uint8)
        self.kernel13 = np.ones((13, 13), np.uint8)
        self.kernel19 = np.ones((19, 19), np.uint8)
        self.kernel25 = np.ones((25, 25), np.uint8)
        self.show = np.zeros(shape=(960, 1280, 3), dtype=np.uint8)
        self.bgTimeCost=0
        #for show

    def avgBackground(self, I):
        # read background pics from disk,and then calculate every frame difference,and accumulate every frame difference
        # #to a sum of frame difference,and then calculate the average frame difference,meanwhile,accumulate every frame to a sum of frame and
        #then calculate the average frame.
        """
        Parameters
        --------------
        img: input  Mat type pic stream

        Returns
        -------

        Examples
        --------
        """
        global IavgF
        global IprevF
        global Iscratch2
        global IdiffF
        global Icount

        cv2.accumulate(I, self.IavgF)
        # cv2.absdiff(I,IprevF, Iscratch2)
        self.Iscratch2 = cv2.absdiff(I, self.IprevF)
        cv2.accumulate(self.Iscratch2, self.IdiffF)

        print("IavgF[100,100,0]:", self.IavgF[100, 100, 0])
        print("IdiffF[100,100,0]:", self.IdiffF[100, 100, 0])
        self.Icount += 1.0
        self.IprevF = I.copy()

    def createModelsfromStats(self, scale):
        # average the frame and frame difference to get the background model
        """
           Parameters
           --------------
           I:      input cam pic waited for segment
           dst:    segment result
           Returns
           -------

           Examples
           --------
           """
        global IavgF
        global IdiffF
        global IhiF
        global IlowF
        #global Icount
        print("Icount", self.Icount)
        # Icount+=1
        self.IavgF = self.IavgF/self.Icount
        self.IdiffF = self.IdiffF/self.Icount
        print("IavgF[100,100,0]:", self.IavgF[100, 100, 0])
        print("IdiffF[100,100,0]:", self.IdiffF[100, 100, 0])
        self.IdiffF = cv2.add(self.IdiffF, 1.0)
        # cv2.imshow("avg", IavgF)
        # cv2.imshow("diff", IdiffF)
        #cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\a.jpg", self.IavgF)
        #cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\d.jpg", self.IdiffF)
        # cv2.add(IavgF,IdiffF, IhiF)

        self.IdiffF = self.IdiffF * scale
        print("IdiffF[mod:", self.IdiffF[100, 100, 0])
        self.IhiF = cv2.add(self.IavgF, self.IdiffF)
        # cv2.subtract(IavgF, IdiffF, IlowF)
        self.IlowF = cv2.subtract(self.IavgF, self.IdiffF)
        #cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\h.jpg", self.IhiF)
        #cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\l.jpg", self.IlowF)

    def studyBackgroundFromCam(self, cam):
        # get 100 pics for time interval of 60sec by cam and save the pics as background pics sets in disk.
        """
            Parameters
             --------------
             cam: input camera object
               Returns
            -------
               Examples
            --------
            """
        global bgVector
        global BG_STUDY_NUM
        try:
            over_flag = 1
            pic_cnt = 0
            while (over_flag):
                frame, nFrameNum = cam.getImage()
                fin = np.float32(frame)
                print("shape", fin.shape)
                self.bgVector[pic_cnt] = fin
                print("pic_cnt", pic_cnt)
                cv2.waitKey(200)
                pic_cnt += 1
                print("pic_cnt", pic_cnt)
                if (pic_cnt == self.BG_STUDY_NUM):
                    over_flag = 0

            print("shapebg", self.bgVector.shape)
            for i in range(self.bgVector.shape[0]):
                print("i", i)
                self.avgBackground(self.bgVector[i])
        except Exception as e:
            print(e)
            cam.destroy()


    def backgroundDiff(self, src0, dst):
        # when get pic frame from camera, use the backgroundDiff to  segment the frame pic;
        # if the pic pixel value is higher than  high background threadhold  or lower than low background threadhold, the pixels
        # will change to white,otherwise, it will cover to black.
        # https://www.cnblogs.com/mrfri/p/8550328.html
        # rectArray=np.zeros(shape=(1,4),dtype=float)

        """
               Parameters
               --------------
               I:      input cam pic waited for segment
               dst:    segment result

               Returns
               -------
               Examples
               --------
           """
        global IlowF
        global IhiF
        global rectArray
        global show
        rectArray = []
        src = np.float32(src0)
        print("IlowF.shape", self.IlowF.shape)
        print("IhiF.shape", self.IhiF.shape)
        print("src.shape", src.shape)
        print("dst.shape", dst.shape)

        print("IlowF.tpye", self.IlowF.dtype)
        print("IhiF.tpye", self.IhiF.dtype)
        print("src.tpye", src.dtype)
        print("dst.tpye", dst.dtype)

        # cv2.inRange(src, IlowF, IhiF, dst)
        dst = cv2.inRange(src, self.IlowF, self.IhiF)
        #cv2.imshow("segment_debug", dst)
        print("is this ok00?")
        dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, self.kernel7)
        print("is this ok01?")
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, self.kernel7)
        print("is this ok02?")

        tmp = 255 * np.ones(shape=dst.shape, dtype=dst.dtype)
        # np.zeros(shape=(960, 1280, 3), dtype=np.uint8)

        dst = cv2.subtract(255, dst)
        # dst=tmp-dst
        print("is this ok03?")
        # cv2.GaussianBlur(dst, dst, (19, 19), 3)
        dst = cv2.GaussianBlur(dst, (19, 19), 3)
        print("is this ok04?")
        dst = cv2.dilate(dst, self.kernel19)
        dst = cv2.dilate(dst, self.kernel19)
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, self.kernel13)  # eclipice
        binary, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(self.show, contours, -1, (0, 255, 0), 3)
        contourLen = len(contours)
        print(contourLen)
        momentList = []
        pointList = []
        if contourLen > 0:  # range create a list of interger ,useful in loop
            for i in range(contourLen):
                contourM = cv2.moments(contours[i])  # every contour's  moment
                contourCenterGx = int(contourM['m10'] / contourM['m00'])
                contourCenterGy = int(contourM['m01'] / contourM['m00'])
                contourArea = cv2.contourArea(contours[i])
                contourhull = cv2.convexHull(contours[i])  # 凸包
                cv2.polylines(self.show, [contourhull], True, (500, 255, 0), 2)

                contourBndBox = cv2.boundingRect(contours[i])  # x,y,w,h
                print("contourBndBox type", type(contourBndBox))
                x = contourBndBox[0]
                y = contourBndBox[1]
                w = contourBndBox[2]
                h = contourBndBox[3]
                img = cv2.rectangle(self.show, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rows, cols = src.shape[:2]  # shape 0 1
                [vx, vy, x, y] = cv2.fitLine(contours[i], cv2.DIST_L2, 0, 0.01, 0.01)
                lefty = int((x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)
                print("pre final")
                # rectArray = np.append(rectArray, contourBndBox, axis=0)
                rectArray.append(contourBndBox)

                print("final")
                print("rectArray", rectArray)
        return rectArray, dst

    def delBg(self,src):
        prev_time = timer()
        #simply output the frame that delete the background
        dst = np.zeros(shape=(960, 1280, 3), dtype=np.uint8)
        resarray, bgMask = self.backgroundDiff(src, dst)
        frame_delimite_bac = cv2.bitwise_and(src, src, mask=bgMask)
        curr_time = timer()
        #calculate the cost time
        exec_time = curr_time - prev_time
        self.bgTimeCost =exec_time
        print("Del background Cost time:", self.bgTimeCost)
        return frame_delimite_bac

    # checkImage

    # Description:
    # checkImage is implemented by sequential procedure, and  theses procedure are  expressed  several functions as below.

    # checkImage Implemente Details:

if __name__ == "__main__":
    # global bgVector
    # global show
    cam = Camera()
    # nConnectionNum = cam.get_device_num()
    #_data_buf, _nPayloadSize = cam.connectCam()
    bgobj = Bglearn()
    bgobj.studyBackgroundFromCam(cam)
    bgobj.createModelsfromStats(6.0)
    while 1:
        try:
            frame, nFrameNum = cam.getImage()
            cv2.imshow("cam", frame)
            bgobj.show = frame.copy()
            #dst = np.zeros(shape=(960, 1280, 3), dtype=np.uint8)
            #resarray, bgMask = bgobj.backgroundDiff(frame, dst)
            #frame_delimite_bac = cv2.bitwise_and(frame, frame, mask=bgMask)


            #cv2.imshow("show0", show)
            #cv2.imshow("show1", bgMask)

            frameDelBg = bgobj.delBg(frame)
            cv2.imshow("output", frameDelBg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.waitKey(10)
        except Exception as e:
            print(e)
            cam.destroy()
    cam.destroy()
