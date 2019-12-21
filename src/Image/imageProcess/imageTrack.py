from src.Image.imageProcess.bgLearn import Bglearn
import cv2
import numpy as np
class ImageTrack:
    def __init__(self):
        self.MAX_CORNERS = 500
        self.win_size = 10
    def LKlightflow_track(self,featureimg,secondimg_orig):
        img_sz = featureimg.shape
        win_size = 10
        drawimg = featureimg.copy()
        drawimg2 = secondimg_orig.copy()
        secondimg =secondimg_orig.copy()
        featureimg = cv2.cvtColor(featureimg,  cv2.COLOR_BGR2GRAY )
        secondimg = cv2.cvtColor(secondimg_orig , cv2.COLOR_BGR2GRAY)
        corner_count = self.MAX_CORNERS
        cornersA = cv2.goodFeaturesToTrack(featureimg, corner_count, 0.01, 5.0)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        cv2.cornerSubPix(featureimg, cornersA, (self.win_size, self.win_size), (-1, -1), criteria)
        #corners_cnt = cornersA.size().height
        #get matrix row num
        corners_cnt = cornersA.shape[0]
        pyramid1 = cv2.buildOpticalFlowPyramid(featureimg, (self.win_size, self.win_size), 3)
        pyramid2 = cv2.buildOpticalFlowPyramid(secondimg,  (self.win_size, self.win_size), 3)
        cornersB = []
        cv2.calcOpticalFlowPyrLK(featureimg, secondimg, cornersA, cornersB)
        for i in range(corners_cnt):
            p0 = cornersA(i][0],cornersA[i][1],
            p1 = cornersA[i][0],cornersA[i][1],
            circle(show, p1, 2, Scalar(0, 0, 255), -1);

            line(show, p0, p1, Scalar(0, 255, 255), 1);

if __name__ == "__main__":
    obj = ImageTrack()
    a = np.array([[1, 2]])
    b = np.array([3, 4])
    corners_cnt = a.shape[1]
    #sz, drawimg = obj.LKlightflow_track(a, b)
    #c = np.zeros(shape=sz, dtype=float)
    print("sz:", corners_cnt)






