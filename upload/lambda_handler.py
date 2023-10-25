import json
from upload import detecting_trash
from ultralytics import YOLO

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

model = YOLO('./29epoch.pt')

def lambda_handler(event, context):
    body = json.loads(event['body'])
    image_url = body['image_url']
    #image_url = event['image_url']

    bboxes = detecting_trash(model, image_url, mapper_cls)
    
    return {
        'statusCode': 200,
        'bboxes': json.dumps(bboxes, ensure_ascii=False)
    }
