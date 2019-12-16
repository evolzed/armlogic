#bgLearn
#Description:
    #Learn the backgroud by pics from cam then get a background model
    #bgLearn is implemented by sequential procedure, and  theses procedure are  expressed  several functions as below.



#implement Details:



def studyBackgroundFromCam(cam): #get the 100 pics for time interval of 60sec by cam and save the pics as background pics sets in disk.
    """
     Parameters


     --------------
     cam: input camera object


       Returns
    -------


       Examples
    --------
    """


def avgBackground(img):#read background pics from disk,and then calculate every frame difference,and accumulate every frame difference
                       #to a sum of frame difference,and then calculate the average frame difference,meanwhile,accumulate every frame to a sum of frame and
                       #then calculate the average frame.
    """
        Parameters
        --------------
        cam: input  Mat type pic stream


          Returns
       -------



          Examples
       --------
    """

def backgroundDiff(src,dst):#when get pic frame from camera, use the backgroundDiff to  segment the frame pic;
        #if the pic pixel value is higher than  high background threadhold  or lower than low background threadhold, the pixels
        #will change to white,otherwise, it will cover to black.

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