import cv2
import numpy as np

#craet 512*512 black image
img = np.zeros((512,512,3),np.uint8)

#draw line
cv2.line(img,(0,0),(512,512),(0,0,255),1,cv2.LINE_AA)
#put txt
cv2.putText(img,'Draw OpenCV Example',(64,500),cv2.FONT_HERSHEY_COMPLEX,1,(125,125,125),1,cv2.LINE_AA)
#draw rect
cv2.rectangle(img,(64,476),(512-64,506),(125,0,125),4,cv2.LINE_4)
#draw circle
cv2.circle(img,(256,256),64,(0,256,0),2,cv2.LINE_AA)

cv2.rectangle(img,(256-64-2,256-64-2),(256+64+2,256+64+2),(125,0,256),2,cv2.LINE_AA)

#draw triangle
triangles = np.array([
    [(256-2, 0), (0, 512-64-4), (512-4, 512-64-4)]])
cv2.polylines(img,triangles,True,(255,0,0),2,cv2.LINE_AA)
#use cv2 display
cv2.imshow('image',img)
k = cv2.waitKey(0)
# wait for ESC key to exit
if k == 27:
    cv2.destroyAllWindows()
# wait for 's' key to save and exit
elif k == ord('s'):
    cv2.imwrite('black.png',img)
    cv2.destroyAllWindows()