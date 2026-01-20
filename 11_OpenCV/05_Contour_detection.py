import cv2 as cv

img = cv.imread('11_OpenCV/data/Photos/cat.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

canny = cv.Canny(gray, 125, 175)


#Contours 
contours, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE,)

cv.imshow('Cat', canny)
cv.waitKey(0)
cv.destroyAllWindows()


