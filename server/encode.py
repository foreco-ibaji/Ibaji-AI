import cv2
import numpy as np
import io
from fastapi import Response

def image_encode(image):
    _, image_serial = cv2.imencode('.png', image)
    image_bytes = io.BytesIO(image_serial.tobytes())
    #image_bytes = image_serial.tobytes()
    
    return image_bytes