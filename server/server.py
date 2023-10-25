from fastapi import FastAPI, UploadFile, File, Response
import uvicorn

from ultralytics import YOLO

from upload import detecting_trash
from mission import get_random_crop_mission_image

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

model = YOLO('./wiz/29epoch.pt')

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/upload")
async def upload(data: dict):
    image_url = data.get('image_url')
    if image_url:
        bboxes = detecting_trash(model, image_url, mapper_cls)
        return {'bboxes': bboxes}
    else:
        return {"error": "Invalid data"}

@app.post("/mission/randomCrop")
async def mission_randomCrop(data: dict):
  image_url = data.get('image_url')
  txt_url = data.get('txt_url')
  if image_url:
    mission_images = get_random_crop_mission_image(image_url, txt_url)
    return {'image': mission_images}
  else:
      return {"error": "Invalid data"}
    
if __name__ == "__main__":
    uvicorn.run(
        app="server:app",
        host="0.0.0.0",
        #host="127.0.0.1",
        port=8080,
        reload=True
    )



