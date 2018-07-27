
import numpy as np
import cv2

img=cv2.imread('arkaplan.jpg', 0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fgbg = cv2.createBackgroundSubtractorMOG2()


cv2.imshow('new',img)
key=cv2.waitKey(0)
cv2.destroyAllWindows()