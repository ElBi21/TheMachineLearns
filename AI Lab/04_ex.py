import cv2

image = cv2.imread("imgs/04_imgs/gerry.png")

ims = cv2.resize(image, None, fx=0.5, fy=0.5)

cv2.imshow("Original Gerry", image)
cv2.waitKey(0)