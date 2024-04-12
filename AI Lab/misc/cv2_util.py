import cv2

def showimgs(imgs: dict):
    """Shows some images in multiple `cv2` windows
    
    Parameters:
        `imgs`: a dictionary where the key is the name of the window and the value is the image"""
    for name in imgs.keys():
        cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(name, imgs[name])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ratioTest(matches, value: float) -> list:
    """Performs the ratio test with a custom value
    
    """
    good_matches = []
    for m, n in matches:
        if m.distance < 0.3 * n.distance:
            good_matches.append(m)
    return good_matches