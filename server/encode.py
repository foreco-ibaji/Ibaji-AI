import cv2
import numpy as np

def image_encode(image):
    _, image_serial = cv2.imencode('.png', image)
    image_bytes = image_serial.tobytes()
    return image_bytes