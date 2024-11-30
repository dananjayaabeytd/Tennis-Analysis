from ultralytics import YOLO

model = YOLO("yolov8x")

model.predict('input_videos/image.png',save=True)
# result = model.track('input_videos/image.png',save=True)