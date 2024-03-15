import numpy as np
import cv2

pts1 = np.float32([[135, 45], [385, 45], [135, 230]])
pts2 = np.float32([[135, 45], [385, 45], [150, 230]])

'''This image takes two images as input and computes the transformation needed to rotate the image'''
transf_matrix = cv2.getAffineTransform(pts1, pts2)

'''With warpAffine() we can transform an image given an affine matrix which describes a transformation.
With this function some properties are preserved from the initial image to the final image:
 - any parallel line will still be parallel (although the angles between the lines will slightly change);
 '''
dest = cv2.warpAffine(pts1, transf_matrix, (800, 800))

print(transf_matrix)

image = cv2.imread("imgs/04_imgs/gerry.png")
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
