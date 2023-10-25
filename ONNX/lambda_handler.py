import onnxruntime
from postprocess import postprocess
from preprocess import preprocess
import requests
import json

mapper_cls = {
    0: '가구',
    1: '고철',
    2: '나무',
    3: '도기',
    4: '비닐',
    5: '스티로폼',
    6: '유리병',
    7: '의류',
    8: '자전거',
    9: '전자제품',
    10: '종이',
    11: '캔',
    12: '페트병',
    13: '플라스틱',
    14: '형광등'
}

def lambda_handler(event, context):
  body = json.loads(event['body'])
  image_url = body['image_url']

  response = requests.get(image_url)
  image_bytes = response.content

  ort_session = onnxruntime.InferenceSession("29epoch.onnx")
  image = preprocess(image_bytes, 640, 640)
  ort_inputs = {ort_session.get_inputs()[0].name: image}
  output = ort_session.run(None, ort_inputs)
  logits = postprocess(image, output, 640, 640, 640, 640, 0.5, 0.5)
  
  bboxes = []
  for logit in logits:
    score, class_id, bbox = logit
    bbox = [mapper_cls[class_id], *bbox]
    bboxes.append(bbox)

  return {
      'statusCode': 200,
      'body': json.dumps({"bboxes" : bboxes})
  }

  