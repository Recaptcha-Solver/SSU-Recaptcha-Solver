import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

import yolov8.yolov8_predict
import yolov8.yolov8_image_utils as yolov8_image_utils

org_image_path = f'trf.jpg'
color = (0, 255, 0)
thickness = 2


model3 = YOLO("yolov8x-seg.pt")
results = model3.predict(org_image_path,save=True, save_txt=True)
# 함수 테스트
yolov8_image_utils.select_16x16(results,cv2.imread(org_image_path),'bus',True)



# 복합
font =  cv2.FONT_HERSHEY_PLAIN

img = cv2.imread(org_image_path)
image_obj = Image.open(org_image_path)  # 한장의 전체 이미지
numpy_image = np.array(image_obj)
class_names = list(map(int,results[0].boxes.cls.numpy()))
# 박스들
for idx,box in enumerate(results[0].boxes):
    x,y,w,h = list(box.xyxy.numpy()[0])
    label_number = class_names[idx] # ex) 0  (0이면 사람, 1이면 ~~)
    name = results[0].names[label_number] # ex) person
    img = cv2.putText(img, name, (int(x), int(y)), font, 2, (0,0,255), 1, cv2.LINE_AA)
    print(f"상자 {idx+1} : {name}")
    cv2.rectangle(img,(int(x), int(y)), (int(w), int(h)), color, thickness)

cv2.imshow('asa',img)
cv2.waitKey(0)
print("\n\n\n")
print(type(results))
print(results)