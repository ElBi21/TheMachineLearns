# Solution proposed by the professor for the homography exercise

import cv2
import numpy as np

prefix = "AI Lab/exercises/01_homography/"

board = cv2.imread(prefix + "billboard.jpg")
board_copy = board.copy()
image = cv2.imread(prefix + "image.jpg")

def onClick(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN: 
        if len(dst_points) < 4:
            dst_points.append([x, y])
            cv2.circle(board_copy, (x, y), 10, (0, 0, 255), -1)
            cv2.imshow("Image", board_copy)

board_h, board_w = board.shape[:2]
image_h, image_w = image.shape[:2]

src_points = np.array([
    [0, 0],
    [0, image_h],
    [image_w, image_h],
    [image_w, 0]
], dtype=np.float32)

dst_points = []


cv2.namedWindow("Image", cv2.WINDOW_FREERATIO)
cv2.setMouseCallback("Image", onClick)
cv2.imshow("Image", board_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(dst_points)
dst_float = np.array(dst_points, dtype=np.float32)
homography_matrix = cv2.getPerspectiveTransform(src_points, dst_float)
warped = cv2.warpPerspective(image, homography_matrix, (board_w, board_h))

