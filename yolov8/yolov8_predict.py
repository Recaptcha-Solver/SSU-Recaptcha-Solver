import cv2
from ultralytics import YOLO

yolov8_model = YOLO("yolov8x-seg.pt")


async def predict_yolov8(image_path):
    global yolov8_model
    results = yolov8_model.predict(image_path, save=True, save_txt=True)
    return results


