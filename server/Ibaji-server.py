from fastapi import FastAPI, UploadFile, File, Response
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
import requests

def Decode(image_bytes, channels=3):
    if channels == 3:
        color = cv2.IMREAD_COLOR
    elif channels == 1:
        color = cv2.IMREAD_GRAYSCALE

    image_serial = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_serial, color)
    return image

# FastAPI 인스턴스 생성
app = FastAPI()

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

model = YOLO('C:\\Users\\juntae\\sesac\\datasets\\aihub\\waste\\runs\\detect\\train22\\epocheee\\29epoch.pt')

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/upload")
async def process_image(data: dict):
    image_url = data.get('image_url')
    if image_url:
        # 이미지 다운로드 및 처리 로직
        response = requests.get(image_url)
        image_bytes = response.content
        image = Decode(image_bytes, channels=3)
        results = model(image)

        if results:
            bboxes = []
            annotations = []

            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().detach().numpy()[0]
                cls = box.cls.cpu().detach().numpy()
                bboxes.append(
                    [mapper_cls[int(cls)], int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))]
                )
                annotations.append({'bboxes': bboxes})

            return {'bboxes': bboxes}
    else:
        return {"error": "Invalid data"}
    
if __name__ == "__main__":
    uvicorn.run(
        app="sesac:app",
        host="0.0.0.0",
        #host="127.0.0.1",
        port=8080,
        reload=True
    )



