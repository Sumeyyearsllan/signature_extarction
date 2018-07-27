import cv2
import numpy as np

imgL = cv2.imread('Tarama.jpg')
img1 = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR = cv2.imread('Tarama2.jpg')
img2 = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoBM_create(numDisparities = 16, blockSize = 17)
disparity = stereo.compute(img2, img1)

cv2.imshow('DepthMap', disparity)
cv2.waitKey()
cv2.destroyAllWindows()