import requests
from decode import image_decode
import cv2
import numpy as np

import albumentations as A

import os

# MPLCONFIGDIR 환경 변수를 설정
os.environ['MPLCONFIGDIR'] = '/tmp'

# YOLO_CONFIG_DIR 환경 변수를 설정
os.environ['YOLO_CONFIG_DIR'] = '/tmp'

def transforming(image):
  transform =  A.Compose([
      A.Resize(width=640, height=640)
  ])
  # Augment an image
  transformed = transform(image=image)
  transformed_image = transformed["image"]
  return transformed_image

def detecting_trash(model, image_url, mapper_cls):
  response = requests.get(image_url)
  image_bytes = response.content
  image = image_decode(image_bytes, channels=3)
  image = transforming(image)
  results = model(image, verbose=False)
  bboxes = []
  
  for result in results:
      boxes = result.boxes  # Boxes object for bbox outputs

  for box in boxes:
      x1, y1, x2, y2 = box.xyxy.cpu().detach().numpy()[0]
      output = box.cls.cpu().detach().numpy()
      bboxes.append(
          [mapper_cls[int(output)], int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))]
      )
  return bboxes