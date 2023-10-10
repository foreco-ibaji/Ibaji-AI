import requests
from decode import image_decode
from encode import image_encode
import cv2
import numpy as np

import albumentations as A

# Declare an augmentation pipeline

def transforming(image, x1, y1, x2, y2):
  transform =  A.Compose([
      A.Crop(x_min=x1, y_min=y1, x_max=x2, y_max=y2, always_apply=True, p=1.0),
      A.RandomCrop(width=256, height=256),
      A.Resize(width=512, height=512)
  ])
  # Augment an image
  transformed = transform(image=image)
  transformed_image = transformed["image"]
  return transformed_image

def get_random_crop_mission_image(image_url, x, y, w, h):
  response = requests.get(image_url)
  image_bytes = response.content
  image = image_decode(image_bytes, channels=3)

  x, y, w, h = map(float, [x, y, w, h])

  # 좌표를 이미지 크기에 맞게 변환
  height, width, _ = image.shape
  x1 = int((x - w / 2) * width)
  y1 = int((y - h / 2) * height)
  x2 = int((x + w / 2) * width)
  y2 = int((y + h / 2) * height)
  transformed_image = transforming(image, x1, y1, x2, y2)
  emcoded_image = image_encode(transformed_image)
  return emcoded_image