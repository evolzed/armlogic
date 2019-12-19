import cv2
import numpy as np
from lib.camera import Camera

# bgLearn
# Description:
# Learn the backgroud by pics from cam then get a background model
# bgLearn is implemented by sequential procedure, and  theses procedure are  expressed  several functions as below.

"""
cam = Camera()
	nConnectionNum = cam.get_device_num()
	_cam, _data_buf, _nPayloadSize = cam.connect_cam()
	while 1:
		frame = cam.getImage(_cam, _data_buf, _nPayloadSize)
		cv2.imshow("cam",frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		cv2.waitKey(10)
"""
BG_STUDY_NUM = 50
bgVector = np.zeros(shape=(BG_STUDY_NUM, 960, 1280, 3), dtype=np.float32)
# bgVector = np.zeros([])
IavgF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)
IprevF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)
Iscratch2 = np.zeros(shape=(960, 1280, 3), dtype=np.float32)
IdiffF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)
IhiF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)
IlowF = np.zeros(shape=(960, 1280, 3), dtype=np.float32)
Icount = 0
kernel5 = np.ones((5, 5), np.uint8)
kernel7 = np.ones((7, 7), np.uint8)
kernel13 = np.ones((13, 13), np.uint8)
kernel19 = np.ones((19, 19), np.uint8)
kernel25 = np.ones((25, 25), np.uint8)
show = np.zeros(shape=(960, 1280, 3), dtype=np.uint8)


def avgBackground(
        I):  # read background pics from disk,and then calculate every frame difference,and accumulate every frame difference#to a sum of frame difference,and then calculate the average frame difference,meanwhile,accumulate every frame to a sum of frame and#then calculate the average frame.
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

    cv2.accumulate(I, IavgF)
    # cv2.absdiff(I,IprevF, Iscratch2)
    Iscratch2 = cv2.absdiff(I, IprevF)
    cv2.accumulate(Iscratch2, IdiffF)

    print("IavgF[100,100,0]:", IavgF[100, 100, 0])
    print("IdiffF[100,100,0]:", IdiffF[100, 100, 0])
    Icount += 1.0
    IprevF = I.copy()


def createModelsfromStats(scale):  # average the frame and frame difference to get the background model
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
    global Icount
    print("Icount", Icount)
    # Icount+=1
    IavgF = IavgF / (Icount)
    IdiffF = IdiffF / (Icount)
    print("IavgF[100,100,0]:", IavgF[100, 100, 0])
    print("IdiffF[100,100,0]:", IdiffF[100, 100, 0])

    # cv2.add(IdiffF, 1.0, IdiffF)  # 这个必须有
    IdiffF = cv2.add(IdiffF, 1.0)  # 这个必须有
    # cv2.imshow("avg", IavgF)
    # cv2.imshow("diff", IdiffF)
    cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\a.jpg", IavgF)
    cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\d.jpg", IdiffF)
    # cv2.add(IavgF,IdiffF, IhiF)

    IdiffF = IdiffF * scale
    print("IdiffF[mod:", IdiffF[100, 100, 0])
    IhiF = cv2.add(IavgF, IdiffF)
    # cv2.subtract(IavgF, IdiffF, IlowF)
    IlowF = cv2.subtract(IavgF, IdiffF)
    cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\h.jpg", IhiF)
    cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\l.jpg", IlowF)


# IavgF.convertTo(IavgF, CV_32FC3, (double)(1.0 / Icount));
# IdiffF.convertTo(IdiffF, CV_32FC3, (double)(1.0 / Icount));


# setHighThreshold(high_threadhold);
# setLowThreshold(low_threadhold);

def studyBackgroundFromCam(
        cam):  # get 100 pics for time interval of 60sec by cam and save the pics as background pics sets in disk.
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
            frame = cam.getImage(_cam, _data_buf, _nPayloadSize)
            fin = np.float32(frame)
            print("shape", fin.shape)
            # bgVector.append(fin)
            # bgVector=np.append(bgVector,fin,axis=0)
            bgVector[pic_cnt] = fin
            print("pic_cnt", pic_cnt)
            cv2.waitKey(200)
            pic_cnt += 1
            print("pic_cnt", pic_cnt)
            if (pic_cnt == BG_STUDY_NUM):
                over_flag = 0

        # learnBackGroundFromVec(bgVector)
        print("shapebg", bgVector.shape)
        for i in range(bgVector.shape[0]):
            # cv2.imshow("cam" + str(i), bgVector[i])
            print("i", i)
            avgBackground(bgVector[i])
    except Exception as e:
        print(e)
        cam.destroy(_cam, _data_buf)


"""
		for i, val in enumerate(bgVector):
			# print(bgVector.index(i))
			print(i)
			#cv2.imshow("cam" + str(i), val)
			print(val.ndim)
			avgBackground(val)
"""

# https://www.cnblogs.com/mrfri/p/8550328.html
# rectArray=np.zeros(shape=(1,4),dtype=float)
rectArray = []


def backgroundDiff(src0, dst):  # when get pic frame from camera, use the backgroundDiff to  segment the frame pic;
    # if the pic pixel value is higher than  high background threadhold  or lower than low background threadhold, the pixels
    # will change to white,otherwise, it will cover to black.

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
    src = np.float32(src0)
    print("IlowF.shape", IlowF.shape)
    print("IhiF.shape", IhiF.shape)
    print("src.shape", src.shape)
    print("dst.shape", dst.shape)

    print("IlowF.tpye", IlowF.dtype)
    print("IhiF.tpye", IhiF.dtype)
    print("src.tpye", src.dtype)
    print("dst.tpye", dst.dtype)

    # cv2.inRange(src, IlowF, IhiF, dst)
    dst = cv2.inRange(src, IlowF, IhiF)
    cv2.imshow("segment_debug", dst)
    print("is this ok00?")
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel7)
    print("is this ok01?")
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel7)
    print("is this ok02?")

    tmp = 255 * np.ones(shape=dst.shape, dtype=dst.dtype)
    # np.zeros(shape=(960, 1280, 3), dtype=np.uint8)

    dst = cv2.subtract(255, dst)
    # dst=tmp-dst
    print("is this ok03?")
    # cv2.GaussianBlur(dst, dst, (19, 19), 3)
    dst = cv2.GaussianBlur(dst, (19, 19), 3)
    print("is this ok04?")
    dst = cv2.dilate(dst, kernel19)
    dst = cv2.dilate(dst, kernel19)
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel13)  # eclipice
    binary, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(show, contours, -1, (0, 255, 0), 3)
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
            cv2.polylines(show, [contourhull], True, (500, 255, 0), 2)

            contourBndBox = cv2.boundingRect(contours[i])  # x,y,w,h
            print("contourBndBox type", type(contourBndBox))
            x = contourBndBox[0]
            y = contourBndBox[1]
            w = contourBndBox[2]
            h = contourBndBox[3]
            img = cv2.rectangle(show, (x, y), (x + w, y + h), (0, 255, 0), 2)
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


# fitLine

# img = cv2.line(src,(cols-1,righty),(0,lefty),(0,255,0),2)

# checkImage

# Description:
# checkImage is implemented by sequential procedure, and  theses procedure are  expressed  several functions as below.

# checkImage Implemente Details:

if __name__ == "__main__":
    # global bgVector
    # global show
    cam = Camera()
    nConnectionNum = cam.get_device_num()
    _cam, _data_buf, _nPayloadSize = cam.connect_cam()
    studyBackgroundFromCam(cam)
    createModelsfromStats(6.0)
    while 1:
        try:
            print("Icount", Icount)
            frame = cam.getImage(_cam, _data_buf, _nPayloadSize)
            cv2.imshow("cam", frame)
            show = frame.copy()
            dst = np.zeros(shape=(960, 1280, 3), dtype=np.uint8)q
            resarray, bgMask = backgroundDiff(frame, dst)
            frame_delimite_bac = cv2.bitwise_and(frame, frame, mask=bgMask)
            cv2.imshow("show0", show)
            cv2.imshow("show1", bgMask)
            cv2.imshow("output",frame_delimite_bac)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.waitKey(10)
        except Exception as e:
            print(e)
            cam.destroy(_cam, _data_buf)

    cv2.waitKey(10)

    cam.destroy(_cam, _data_buf)
    """
    while 1:
        frame = cam.getImage(_cam, _data_buf, _nPayloadSize)
        cv2.imshow("cam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.waitKey(10)
    """
