import cv2
import numpy as np

def image_decode(image_bytes, channels=3):
    if channels == 3:
        color = cv2.IMREAD_COLOR
    elif channels == 1:
        color = cv2.IMREAD_GRAYSCALE

    image_serial = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_serial, color)
    return image