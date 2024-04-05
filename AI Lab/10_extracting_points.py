import cv2
import numpy as np

image = cv2.imread("AI Lab/cat.jpeg")

# First method: SIFT (Scale Invariant Feature Transform)

# sift = cv2.SIFT_create()
# keypoints, descriptors = sift.detectAndCompute(image, None)
# cv2.drawKeypoints(image, keypoints, image, (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#cv2.imshow("Image", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Second method: SURF, optimized version of SIFT
# It's patented

# Third method: A-Kaze
akaze = cv2.AKAZE_create()
keypoints, descriptors = akaze.detectAndCompute(image, None)
# cv2.drawKeypoints(image, keypoints, image, (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# We can try to match two images:
image2 = cv2.imread("AI Lab/onboard.png")

keypoints2, descriptors2 = akaze.detectAndCompute(image2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.match(descriptors, descriptors2)

image_matches = cv2.drawMatches(image, keypoints, image2, keypoints2, matches, image2, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


cv2.imshow("Image", image_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()