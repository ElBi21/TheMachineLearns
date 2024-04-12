import cv2
import numpy as np

# Load the images
image1 = cv2.imread("AI Lab/imgs/10_imgs/right.png")
image2 = cv2.imread("AI Lab/imgs/10_imgs/left.png")

# Create featurer extractor
orb = cv2.ORB_create()

# Compute features
kpt1, desc1 = orb.detectAndCompute(image1, None)
kpt2, desc2 = orb.detectAndCompute(image2, None)

# Create a matcher and match the keypoints
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = matcher.knnMatch(desc1, desc2, k=2)

# Perform the ratio test: a match is said to be correct if the ratio between the two
# closest points is below a certain threshold
good_matches = []
for m, n in matches:
    if m.distance < 0.3 * n.distance:
        good_matches.append(m)

# Check if at least 4 points have been selected
# Remember that the two images are called queryImg and TrainingImg, so queryIdx is a point
# belonging to queryImg, while TrainingIdx is a point belonging to TrainingImg
if len(good_matches) > 4:
    # Convert to float32
    src_points = np.float32([kpt1[m.queryIdx].pt for m in good_matches])
    dst_points = np.float32([kpt2[m.trainIdx].pt for m in good_matches])

    # Compute the homography matrix
    M, mask = cv2.findHomography(src_points, dst_points)

    print(M)

    # Transform left image and stitch it together with the right image       # \/ this removes the black part
    dst = cv2.warpPerspective(image1, M, (image1.shape[1] + image2.shape[1] - int(M[0, 2]), image1.shape[0]))

    dst[0:image2.shape[0], 0:image2.shape[1]] = image2.copy()

    cv2.namedWindow("Panorama", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Panorama", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()