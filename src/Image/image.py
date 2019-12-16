#import cv2
import numpy as np
#bgLearn
#Description:
#Learn the backgroud by pics from cam then get a background model
#bgLearn is implemented by sequential procedure, and  theses procedure are  expressed  several functions as below.

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
    bgVector=[]
    BGDIR = "E:\\Xscx2019\\OPENCV_PROJ\\backgroundLearn\\"
    over_flag = 1
    BG_STUDY_NUM=10
    while (over_flag):
        pic_cnt = 1
        #Mat frame, blob;
        frame = cam.getImage()
        #Mat fin = Mat::zeros(frame.size(), CV_32FC3);
        fin=cv.zeros(frame.size(), CV_32FC3)
        print(np.array(pin))
        frame.convertTo(fin, CV_32FC3)
        bgVector.append(fin)
        picname = to_string(pic_cnt)
        picname = BGDIR + picname + ".jpg"
        print(picname)
        cv.waitKey(200)
        pic_cnt=pic_cnt+1
        if (pic_cnt == BG_STUDY_NUM):
            over_flag = 0

    #learnBackGroundFromVec(bgVector);

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




#checkImage

#Description:
#checkImage is implemented by sequential procedure, and  theses procedure are  expressed  several functions as below.

#checkImage Implemente Details: