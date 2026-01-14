import cv2 as cv

cap = cv.VideoCapture('11_OpenCV/data/Videos/dog.mp4')

while True:
    ret, frame = cap.read()

    if not ret:
        print('Could not read frame')
        break
    
    cv.imshow('Video', frame)
    if cv.waitKey(1) & 0xff == ord('q'):
        print('Quitting...')
        break

cap.release()
cv.destroyAllWindows()

        

