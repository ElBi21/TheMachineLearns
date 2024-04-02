import numpy as np
import cv2

prefix = "AI Lab/exercises/01_homography/"

board = cv2.imread(prefix + "billboard.jpg")    # Load board
image = cv2.imread(prefix + "Shrek.png")        # Load image

angles = np.array([                             # Save angles
    [305, 452],   # T-Left
    [308, 2385],  # B-Left
    [2662, 2060], # B-Right
    [2672, 748]  # T-Right
], dtype=np.float32)

source_pts = np.array([                         # Get shape of the image
    [0, 0],
    [0, image.shape[0]],
    [image.shape[1], image.shape[0]],
    [image.shape[1], 0]
], dtype=np.float32)

transformation_matrix = cv2.getPerspectiveTransform(source_pts, angles) # Get the transformation matrix
out = cv2.warpPerspective(image, transformation_matrix, (board.shape[1], board.shape[0]))   # Transform the image

# Specifying data types breaks the code: why?
angles = np.array([                             # Save angles
    [305, 452],   # T-Left
    [308, 2385],  # B-Left
    [2662, 2060], # B-Right
    [2672, 748]  # T-Right
])

cv2.fillConvexPoly(board, angles, (0, 0, 0))    # Draw black mask on top of billboard
bit_or = cv2.bitwise_or(board, out)             # Do bitwise or between billboard and out

# Print board
final_r = cv2.resize(bit_or, (960, 540))        # Resize board

cv2.imshow("Final", final_r)
cv2.waitKey(0)
cv2.destroyAllWindows()