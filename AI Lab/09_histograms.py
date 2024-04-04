import cv2
import numpy as np
import matplotlib.pyplot as plt

prefix = "AI Lab/"

image = cv2.imread(prefix + "imgs/04_imgs/gerry.png")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_eq = cv2.equalizeHist(image_gray)

cv2.imwrite(prefix + "06_imgs/gerry_gray.png", image_gray)
cv2.imwrite(prefix + "06_imgs/gerry_gray_eq.png", gray_eq)

channels = cv2.split(image)
eq_channels = []

for chann in channels:
    eq_channels.append(cv2.equalizeHist(chann))

equalized = cv2.merge(eq_channels)

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hue, saturation, value = cv2.split(hsv_image)

equalized_value = cv2.equalizeHist(value)
equalized = cv2.merge([hue, saturation, equalized_value])
equalized = cv2.cvtColor(equalized, cv2.COLOR_HSV2RGB)

# Create an instance of CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

equalized = clahe.apply(image_gray)

#cv2.imshow("Hue", hue)
#cv2.imshow("Saturation", saturation)
#cv2.imshow("Value", equalized_value)
cv2.imshow("Equalized", equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imshow("Gray image", image_gray)
#cv2.imshow("Equalized", gray_eq)
#cv2.imshow("Equalized", equalized)
#cv2.waitKey(0)
#cv2.destroyAllWindows()