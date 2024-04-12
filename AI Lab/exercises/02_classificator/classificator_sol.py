import cv2
import numpy as np

# Load images
imgs = []

prefix = "AI Lab/exercises/02_classificator/"
img1 = cv2.imread(prefix + "db/sw_zelda_totk.jpg")
img2 = cv2.imread(prefix + "db/sw_p5s.jpg")
img3 = cv2.imread(prefix + "db/sw_mk8d.jpg")

classes = ["Zelda TotK", "P5S", "MK 8 Deluxe"]

def descriptorsDB(images):
    descriptors_list = []

    orb = cv2.ORB_create()
    for img in images:
        kpt, des = orb.detectAndCompute(img, None)
        descriptors_list.append(des)

    return descriptors_list

