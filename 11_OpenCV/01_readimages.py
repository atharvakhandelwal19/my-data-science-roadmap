import cv2 as cv

### Opening and displaying the images

img = cv.imread('11_OpenCV/data/Photos/cat.jpg')

# cv.imshow('Cat', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

### Savingtheimage

#cv.imwrite('output.jpg', img)

### Image Dimension

h, w, c = img.shape
print(f'Image Loaded: \nHeight: {h}, Width:{w}, Colorscale: {c }')

### Converting Images to Grayscale

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)
# cv.waitKey(0)
# cv.destroyAllWindows()

