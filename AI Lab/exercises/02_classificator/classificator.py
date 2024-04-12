import numpy as np
import cv2

def showimgs(imgs: dict) -> None:
    """Shows some images in multiple `cv2` windows
    
    Parameters:
        `imgs`: a dictionary where the key is the name of the window and the value is the image"""
    for name in imgs.keys():
        cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(name, imgs[name])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ratioTest(matches, value=0.3) -> list:
    """Performs the ratio test with a custom value. The ratio test considers a match as
    a good match if the following condition holds:
    
    ```python
        matches[0].distance < value * matches[1].distance
            
    ```

    Parameters:
        `matches`: a list of matches between two images;
        `value`: the value that 
    """
    return [m for m, n in matches if m.distance < value * n.distance]


# Import images from db
prefix = "AI Lab/exercises/02_classificator/"

database = {
    "Mario Kart 8 Deluxe": cv2.imread(prefix + "db/sw_mk8d.jpg"),
    "Persona 5 Strikers": cv2.imread(prefix + "db/sw_p5s.jpg"),
    "Zelda Tears of the Kingdom": cv2.imread(prefix + "db/sw_zelda_totk.jpg")
}

# Create feature extractor
akaze = cv2.AKAZE_create()

# List with all keypoints and descriptors of the images of the db
# Stored as:
#       [      (0)         (1)
#           [keypts1, descriptor1], (0) MK8D
#           [keypts2, descriptor2], (1) P5S
#           [keypts3, descriptor3], (2) ZELDA TOTK
#       ]
db = [akaze.detectAndCompute(img, None) for img in list(database.values())]

# Load a random image
randindex = np.random.randint(3)
to_test = cv2.imread(prefix + "to_classify/" + ["IMG_1889.jpg", "IMG_1890.jpg", "IMG_1891.jpg"][randindex])

# Extract keypoints and descriptor
test_kpts, test_desc = akaze.detectAndCompute(to_test, None)

# Create matcher
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = []

for img in db:
    test_match = matcher.knnMatch(img[1], test_desc, k=2)
    # Ratio test
    matches.append(ratioTest(test_match))

best_match = matches.index(max(matches, key=len))

print(f"The provided image seems similar to {list(database.keys())[best_match]}")
# showimgs({"Original": to_test, "From the database": list(database.values())[best_match]})