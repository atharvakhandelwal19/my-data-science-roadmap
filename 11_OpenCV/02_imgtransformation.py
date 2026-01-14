import cv2 as cv

img = cv.imread('11_OpenCV/data/Photos/cat.jpg')

def rescale(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dim = (width, height)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)

cv.imshow('Cat', img)
resized = rescale(img)
cv.imshow('Resized', resized)

#Reading videos

# capture = cv.VideoCapture('11_OpenCV/data/Videos/dog.mp4')

# while True:
#     isTrue, frame = capture.read()
#     frame_resized = rescale(frame)

#     cv.imshow('original', frame)
#     cv.imshow('resized', frame_resized) 

#     if cv.waitKey(20) & 0xff==ord('d'):
#         break

cv.waitKey(0)
cv.destroyAllWindows()