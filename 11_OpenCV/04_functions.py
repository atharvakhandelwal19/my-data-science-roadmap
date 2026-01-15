import cv2 as cv

img = cv.imread('11_OpenCV/data/Photos/cat.jpg')

# Converting to GrayScale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Blur
blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)

#Edge Cascade
canny = cv.Canny(img,125,175)

#Dilating the image
dilated = cv.dilate(canny, (3,3), iterations=1)

#Eroding the image
eroded = cv.erode(dilated,(3,3), iterations=1)
cv.imshow('Cat', eroded)
cv.waitKey(0)
cv.destroyAllWindows()


