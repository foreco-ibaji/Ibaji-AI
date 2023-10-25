import cv2
import numpy as np
import io
import base64

def image_encode(image):
    _, image_serial = cv2.imencode('.png', image)
    #image_bytes = io.BytesIO(image_serial.tobytes())
    image_bytes = image_serial.tobytes()
    encoded_string = base64.b64encode(image_bytes).decode()
    
    return encoded_string