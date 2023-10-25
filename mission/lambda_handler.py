import json
from mission import get_random_crop_mission_image

def lambda_handler(event, context):
    body = json.loads(event['body'])
    image_url = body['image_url']
    coordinate = body['coordinate']

    mission_images = get_random_crop_mission_image(image_url, coordinate)

    return {
      'statusCode': 200,
      'body': json.dumps({"images" : mission_images})
  }