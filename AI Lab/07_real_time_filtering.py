import cv2
import numpy as np

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    # Bring the image to grayscale
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Soft clean the image
    image_gray = cv2.medianBlur(image_gray, 5)

    # Use the Laplacian filter in order to extract the contours
    edges = cv2.Laplacian(image_gray, cv2.CV_8U, ksize=5)

    # Threshold the edges in order to get only some valid edges (so with value greater than 70)
    ret, threshold = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)

    # Now we can extract the colors. We can use the bilateral filter with high values, so that we can keep some edges
    color_image = cv2.bilateralFilter(frame, 10, 250, 250)

    # Put together the two images, so the color and the sketch
    sketch = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)

    # Do the bitwise of the two images and merge the sketch and the color
    final_image = cv2.bitwise_and(color_image, sketch)

    cv2.imshow("Output of camera", final_image)
    k = cv2.waitKey(10)

    if k == ord("q"):
        break