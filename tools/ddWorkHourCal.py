import cv2
import numpy as np
from PIL import ImageGrab
import numpy
import time
from timeit import default_timer as timer


def capture(left, top, right, bottom):
    # img = ImageGrab.grab(bbox=(left, top, right, bottom))
    img = ImageGrab.grab() # full screen
    img = np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
    r, g, b = cv2.split(img)
    cv2.merge([b, g, r], img)
    return img

prev_time = timer()
prev_cap =capture(0, 0, 600, 400)
x = 2
working_threshold = 1000.0/2073600.0
work_hours = 0
while 1:
    curr_time = timer()
    randomInterval = numpy.random.uniform(low=0.5 * x, high=2.0 * x, size=1)
    if curr_time - prev_time >= randomInterval:
        print("in")
        curr_cap = capture(0, 0, 600, 400)
        diff = cv2.absdiff(curr_cap, prev_cap)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diffSize = diff[diff == 255].size
        totalSize = diff.size
        thresh = float(diffSize)/float(totalSize)
        print("diffSize", diffSize)
        print("thresh", thresh)
        print("area", diff.size)
        print(1000.0 / 2073600.0)
        if thresh > working_threshold:
        # if diffSize > 1000:
            work_hours += randomInterval
            print("work_hours", work_hours)
            print("working!!!!")

        cv2.putText(diff, text=str(work_hours), org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.imshow("diff", diff)
        cv2.waitKey(20) #要抑制速度
        prev_time = curr_time
        prev_cap = curr_cap


    # cv2.waitKey(0)
