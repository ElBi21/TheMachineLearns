import cv2
import numpy as np

# Consider two pixels, and suppose that we want to do x + y
x = np.uint8([250])
y = np.uint8([50])

result_opencv = cv2.add(x, y)
result_np = x + y

