# Wiz AI
- ### When the user captures an image of recyclable waste using a camera and sends it, the YOLO model detects the recyclable waste from the photo and provides information about its type and location.
- ### We have trained YOLOv8 utilizing the recycled garbage data from AI hub.
- ### In order to enable rapid operation with small memory on the Lambda server, it was deployed in ONNX format.
- ### At present, there are 15 types of recyclable waste that can be detected.
  - ### These include furniture, scrap metal, wood, ceramics, vinyl, styrofoam, glass bottles, clothing, bicycles, electronics, paper, cans, PET bottles, plastic, and fluorescent lights.
- ### We developed an API using AWS Lambda, and resolved the cold start issue by setting the Amazon EventBridge Scheduler to invoke the lambda function every five minutes.

## AI Server pipeline
<img src="./resources/wiz_ai_pipeline.png" width="100%">

## plans
- Construction of an automated process for AI model updates (CI/CD).
