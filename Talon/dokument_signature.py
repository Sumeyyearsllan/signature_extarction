import numpy as np
import cv2

class Rect:
    def __init__(self, x = 0, y = 0, w = 0, h = 0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = 0
    def setArea(self, area):
        self.area = area
    def getArea(self):
        return self.area
    def set(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = w * h
    def addPadding(self, imgSize, padding):
        self.x -= padding
        self.y -= padding
        self.w += 2 * padding
        self.h += 2 * padding
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.x + self.w > imgSize[0]:
            self.w = imgSize[0] - self.x
        if self.y + self.h > imgSize[1]:
            self.h = imgSize[1] - self.y
signature = cv2.imread('C:\Python\Python37\images\yeni4.jpg' , 0)
def getPageFromImage(img):
    imgSize = np.shape(img)#görüntünün özelliklerine ulaşabilmek için shape kullandık.
    cImg = cv2.medianBlur(src=signature, ksize=21) #kenarları korumak için bu metodu kullanrak görüntüyü bulanıklaştırır.
    bImg = signature.copy()#kopyasını alarak imza dosyasının yerini gösterecek.
    threshold, _ = cv2.threshold(src = bImg, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print(threshold)
    cannyImg = cv2.Canny(image = bImg, threshold1 = 0.5* threshold, threshold2 = threshold)
    print(cannyImg)
    _, contours, _ = cv2.findContours(image = cannyImg.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print ('No Page Found')
        return img
    maxRect = Rect(0, 0, 0, 0)
    coordinates = []
    for contour in contours:
        epsilon = cv2.arcLength(contour, True)#true olana kadar bu döngü döner.
        corners = cv2.approxPolyDP(contour, 0.1 * epsilon, True)
        print(corners)
        x, y, w, h = cv2.boundingRect(points = contour)
        currentArea = w * h
        if len(corners) == 4 and currentArea > maxRect.getArea():
            maxRect.set(x, y, w, h)
            print (cv2.isContourConvex(contour)) #eğrinin dış bükey olup olmadığını döndürür.
    contoursInPage = 0
    for contour in contours:
        x, y, _, _ = cv2.boundingRect(points = contour)
        if (x > maxRect.x and x < maxRect.x + maxRect.w) and (y > maxRect.y and y < maxRect.y + maxRect.h):
                contoursInPage += 1
    maxContours = 5
    if contoursInPage <= maxContours:
        print ('No Page Found')
        return img
    return img[maxRect.y : maxRect.y + maxRect.h, maxRect.x : maxRect.x + maxRect.w]
def getSignatureFromPage(img):
    imgSize = np.shape(img)#özellikleri alındı.
    threshold, _ = cv2.threshold(src = signature, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cannyImg = cv2.Canny(image = signature, threshold1 = 0.5 * threshold, threshold2 = threshold)
    kernel = cv2.getStructuringElement(shape = cv2.MORPH_RECT, ksize = (30, 1))
    cannyImg = cv2.morphologyEx(src = cannyImg, op = cv2.MORPH_CLOSE, kernel = kernel)
    _, contours, _ = cv2.findContours(image = cannyImg.copy(), mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    maxRect = Rect(0, 0, 0, 0)
    maxCorners = 0
    for contour in contours:
        epsilon = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.02* epsilon, True)
        x, y, w, h = cv2.boundingRect(points = contour)
        currentArea = w * h
        if len(corners) > maxCorners:
            maxCorners = len(corners)
            maxRect.set(x, y, w, h)
    maxRect.addPadding(imgSize = imgSize, padding = 10)
    return img[maxRect.y : maxRect.y + maxRect.h, maxRect.x : maxRect.x + maxRect.w]

def getSignature(img):
    imgSize = np.shape(img)
    blockSize = 21
    C = 10
    if blockSize > imgSize[0]:
        if imgSize[0] % 2 == 0:
            blockSize = imgSize[0] - 1
        else:
            blockSize = imgSize[0]
    if blockSize > imgSize[1]:
        if imgSize[0] % 2 == 0:
            blockSize = imgSize[1] - 1
        else:
            blockSize = imgSize[1]
    mask = cv2.adaptiveThreshold(signature, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = blockSize, C = C)
    rmask = cv2.bitwise_not(mask)
    return cv2.bitwise_and(signature, signature, mask=rmask)
signature = getPageFromImage(img = signature)
signature = getSignatureFromPage(img = signature)
signature = getSignature(img = signature)
cv2.imshow('Signature', signature)

key = cv2.waitKey(0)