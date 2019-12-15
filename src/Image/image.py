#bgLearn
#Description:
    #Learn the backgroud by pics from cam then get a background model
    #bgLearn is implemented by sequential procedure, and  theses procedure are  expressed  several functions as below.
    #input Parameter

    #output Parameter

    #return



#implemente Details:
    #function:
        # studyBackgroundFromCam
    #Description:
        #get the 100 pics for time interval of 60sec by cam and save the pics as background pics sets in disk.
    #input Parameter

    #output Parameter

    #return




    #function:
        #avgBackground
    #description:
        #read background pics from disk,and then calculate every frame difference,and accumulate every frame difference
        #to a sum of frame difference,and then calculate the average frame difference,meanwhile,accumulate every frame to a sum of frame and
        #then calculate the average frame.
    # input Parameter

    # output Parameter

    # return



    # function:
        #createModelsfromStats
    #description:
        #calculate the BackgroundModel,whichi is composed of  high background threadhold and low background threadhold.
        #add the average frame difference to the  the average frame to make the high background threadhold,
        #subtract the average frame difference from  the  the average frame to make the low background threadhold,
    # input Parameter

    # output Parameter

    # return



    # function:
        #backgroundDiff
    #description:
        #when get pic frame from camera, use the backgroundDiff to  segment the frame pic;
        #if the pic pixel value is higher than  high background threadhold  or lower than low background threadhold, the pixels
        #will change to white,otherwise, it will cover to black.
    # input Parameter

    # output Parameter

    # return



#checkImage

#Description:
    #checkImage is implemented by sequential procedure, and  theses procedure are  expressed  several functions as below.

    #checkImage Implemente Details: