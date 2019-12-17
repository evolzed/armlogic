import cv2
import numpy as np
from lib.camera import Camera
#bgLearn
#Description:
#Learn the backgroud by pics from cam then get a background model
#bgLearn is implemented by sequential procedure, and  theses procedure are  expressed  several functions as below.

bgVector = np.zeros(shape=(10,960,1280,3),dtype=np.float32)
#bgVector = np.zeros([])
IavgF=np.zeros(shape=(960,1280,3),dtype=np.float32)
IprevF=np.zeros(shape=(960,1280,3),dtype=np.float32)
Iscratch2=np.zeros(shape=(960,1280,3),dtype=np.float32)
IdiffF=np.zeros(shape=(960,1280,3),dtype=np.float32)
Icount=0
kernel5 = np.ones((5, 5), np.uint8)
kernel7 = np.ones((7, 7), np.uint8)
kernel13 = np.ones((13, 13), np.uint8)
kernel19 = np.ones((19, 19), np.uint8)
kernel25 = np.ones((25, 25), np.uint8)
show=np.zeros(shape=(960,1280,3),dtype=np.uint8)

def studyBackgroundFromCam(cam): #get 100 pics for time interval of 60sec by cam and save the pics as background pics sets in disk.
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
    try:
        over_flag = 1
        BG_STUDY_NUM = 10
        pic_cnt = 0
        while (over_flag):
            frame = cam.getImage(_cam, _data_buf, _nPayloadSize)
            fin = np.float32(frame)
            print("shape", fin.shape)
            # bgVector.append(fin)
            # bgVector=np.append(bgVector,fin,axis=0)
            bgVector[pic_cnt] = fin
            cv2.waitKey(200)
            pic_cnt += 1
            print(pic_cnt)
            if (pic_cnt == BG_STUDY_NUM):
                over_flag = 0

        # learnBackGroundFromVec(bgVector)
        print("shapebg", bgVector.shape)
        for i in range(bgVector.shape[0]):
            # cv2.imshow("cam" + str(i), bgVector[i])
            avgBackground(bgVector[i])
    except Exception as e:
        print(e)
        cam.destroy(_cam, _data_buf)



def avgBackground(img):#read background pics from disk,and then calculate every frame difference,and accumulate every frame difference
                   #to a sum of frame difference,and then calculate the average frame difference,meanwhile,accumulate every frame to a sum of frame and
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

    cv2.accumulate(I, IavgF)
    cv2.absdiff(I,IprevF, Iscratch2)
    cv2.accumulate(Iscratch2, IdiffF)
    Icount += 1.0
    IprevF = I.copy()



def createModelsfromStats():  # average the frame and frame difference to get the background model
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
    IavgF = IavgF / Icount
    IdiffF = IdiffF / Icount
    cv2.add(IdiffF, 1.0, IdiffF)  # 这个必须有
    cv2.imshow("avg", IavgF)
    cv2.imshow("diff", IdiffF)
    cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\a.jpg", IavgF)
    cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\d.jpg", IdiffF)
    cv2.add(IavgF, IdiffF, IhiF)
    cv2.subtract(IavgF, IdiffF, IlowF)
    cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\h.jpg", IhiF)
    cv2.imwrite("E:\\Xscx2019\\OPENCV_PROJ\\backgroundtemplate\\py\\l.jpg", IlowF)



def backgroundDiff(src,dst):# when get pic frame from camera, use the backgroundDiff to  segment the frame pic;
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
    cv2.inRange(src, IlowF, IhiF, dst)
    cv2.morphologyEx(dst, dst, cv2.MORPH_OPEN, kernel7)
    cv2.morphologyEx(dst, dst, cv2.MORPH_CLOSE, kernel7)
    cv2.subtract(255, dst, dst);
    cv2.GaussianBlur(dst, dst, (19, 19), 3)
    cv2.dilate(dst, dst, kernel19)
    cv2.dilate(dst, dst, kernel19)
    cv2.morphologyEx(dst, dst, cv2.MORPH_CLOSE, kernel13)  #eclipice
    binary,contours, hierarchy  = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(show, contours, -1, (0, 255, 0), 3)
    contourLen=len(contours)
    printf(contourLen)
    if len>0:



#checkImage

#Description:
#checkImage is implemented by sequential procedure, and  theses procedure are  expressed  several functions as below.

#checkImage Implemente Details:


