import cv2
import numpy as np

# Load images
imgs = []

prefix = "AI Lab/exercises/02_classificator/"
img1 = cv2.imread(prefix + "db/sw_zelda_totk.jpg")
img2 = cv2.imread(prefix + "db/sw_p5s.jpg")
img3 = cv2.imread(prefix + "db/sw_mk8d.jpg")
img4 = cv2.imread(prefix + "db/sapienza.jpg")
img5 = cv2.imread(prefix + "db/ale.png")
images = [img1, img2, img3, # img4,
          img5]

classes = ["Zelda TotK", "P5S", "MK 8 Deluxe", #"Pazienza Sapienza",
           "Alessandro Slovenia"]
orb = cv2.ORB_create()

def descriptorsDB(images):
    descriptors_list = []
    
    for img in images:
        kpt, des = orb.detectAndCompute(img, None)
        descriptors_list.append(des)

    return descriptors_list

def objClassification(img, description_list):
    kpt, des = orb.detectAndCompute(img, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    best_matches = []
    class_ID = -1
    for descr in description_list:
        matches = matcher.knnMatch(des, descr, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.2 * n.distance:
                good_matches.append(m)
        best_matches.append(good_matches)
    
    if len(best_matches) > 0:
        max_val = max(best_matches, key=len)
        class_ID = best_matches.index(max_val)

    return class_ID

descriptor_list = descriptorsDB(images)
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    class_ID = objClassification(frame, descriptor_list)

    if class_ID != -1:
        cv2.putText(frame, classes[class_ID], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 4)
        cv2.imshow("Classification", frame)
        k = cv2.waitKey(10)

        if k == ord("q"):
            break
