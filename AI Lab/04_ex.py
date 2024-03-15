import cv2
import numpy as np

# Define the callback function which will be used later on:
def onClick(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN: 
        if len(src_points) < 4:
            src_points.append([x, y])
            cv2.circle(image_copy, (x, y), 10, (0, 0, 255), -1)
            cv2.imshow("Image", image_copy)

image = cv2.imread("AI Lab/imgs/04_imgs/gerry.png")

# Create a copy of the image:
image_copy = image.copy()

# Define the starting point
src_points = []

# Define the destination points
dest_points = np.array([
    [0, 0],
    [0, 800],
    [600, 800],
    [600, 0]
], np.float32)

# Now we have to create a window on which we can use the cursor in order to click:
cv2.namedWindow("Image", cv2.WINDOW_FREERATIO)

# We need to create a callback function, aka a function that gets called on certain, predefined actions.
cv2.setMouseCallback("Image", onClick)

# Now we can proceed after the callback: we show now the image:
cv2.imshow("Image", image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the array into an np.array():
src_float = np.array(src_points, dtype=np.float32)
# Get the transformation matrix
trans_matrix = cv2.getPerspectiveTransform(src_float, dest_points)

out_img = cv2.warpPerspective(image, trans_matrix, (600, 800))

cv2.imshow("Output Image", out_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("AI Lab/imgs/04_imgs/gerry_transf.png", out_img)