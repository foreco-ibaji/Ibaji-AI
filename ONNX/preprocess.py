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

def preprocess(input_image, input_width, input_height):
    """
    Preprocesses the input image before performing inference.

    Returns:
        image_data: Preprocessed image data ready for inference.
    """
    # Read the input image using OpenCV
    #img = cv2.imread(input_image)
    img = image_decode(input_image)

    # Get the height and width of the input image
    img_height, img_width = img.shape[:2]

    # Convert the image color space from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to match the input shape
    img = cv2.resize(img, (input_width, input_height))

    # Normalize the image data by dividing it by 255.0
    image_data = np.array(img) / 255.0

    # Transpose the image to have the channel dimension as the first dimension
    image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

    # Expand the dimensions of the image data to match the expected input shape
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

    # Return the preprocessed image data
    return image_data