import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('Signature.png',0)
img=cv2.bilateralFilter(img, 50, 90,100)
#img=cv2.medianBlur(img, 9) #mühür kayboldu ama imzada bozulma gerçekleşti.
#bImg = img.copy()
#threshold, _ = cv2.threshold(src = bImg, thresh =127, maxval = 255, type = cv2.THRESH_BINARY )


ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','TOZERO_INV']
images = [img,  thresh5]

for i in range(2):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
#kernel=np.ones((3,3), np.int8)/9
#smooted=cv2.filter2D(img, 13, kernel)
#blur=cv2.GaussianBlur(img, (7,7),0)
#median=cv2.medianBlur(img, 5)
#bileteral=cv2.bilateralFilter(img, 5, 20 ,25)

cv2.imshow('bulanık', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
