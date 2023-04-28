import cv2
import numpy as np
from PIL import Image


__all__ = [
    'is_marked',
    "get_output_layers",
    "draw_prediction",
    "predict"
]

import setting
import solverecaptchas.utils as utils


def is_marked(img_path):
    """Detect specific color for detect marked"""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            if r == 0 and g == 0 and b == 254:  # Detect Blue Color
                return True
    return False


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, x, y, x_plus_w, y_plus_h):
    """Paint Rectangle Blue for detect prediction"""
    color = 256  # Blue
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, cv2.FILLED)


async def predict_yolov8():
    pass

async def predict_yolov3(net, file,obj=None):
    """Predict Object on image"""
    file_names = setting.yolov3_txt_path
    image = cv2.imread(file)
    width = image.shape[1]
    height = image.shape[0]
    scale = 0.00392

    conf_threshold = 0.5
    nms_threshold = 0.4

    with open(file_names, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    if obj is None:
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))
        classes_names = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    classes_names.append(classes[class_id])
        return classes_names  # Return all names object in the images
    else:
        out_path = f"pictures/tmp/{hash(file)}.jpg"
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))
        print(f"outs 리스트 : {outs}")

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        out = False
        for i in indices:
            if classes[int(class_ids[int(i)])] == obj or (obj == 'vehicles' and (
                    classes[int(class_ids[int(i)])] == 'car' or classes[int(class_ids[int(i)])] == 'truck')):
                out = out_path
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                draw_prediction(image, round(x), round(y), round(x + w), round(y + h))
            # Save Image
        if out:
            cv2.imwrite(out_path, image)
        return out  # Return path of images or False if not found object


async def predict_tensorflow(file_path,title,tf_net):
    threshold = 0.5 # 0.5 넘으면 포함된다고 인식
    model = tf_net
    image = utils.load_img_to_96x96x1(file_path)
    classes = model.predict(image)
    # best_class = np.argmax(classes) # print(f'클래스 예측 : {best_class}')
    tf_labels_ignore_case = list(map(str.lower ,setting.tf_labels)) # 소문자로

    title_label_num = None # title이 Bus면 몇번 레이블이 Bus인지
    if not (title in tf_labels_ignore_case):
        print(f'{title}은 학습된 텐서플로우 모델에 존재하지 않는 레이블')
        exit(1)
    else:
        title_label_num = tf_labels_ignore_case.index(title)

    return_classes = []
    for idx,predict_result in enumerate(classes[0]):
        if predict_result >= threshold:
            return_classes.append(tf_labels_ignore_case[idx])

    return return_classes


async def predict(net,tf_net, file, title,obj=None):
    title = title.lower()
    labels = list(map(str.lower,setting.use_tf_model_label))
    # 사용할 모델 선택
    if title in labels:
        return await predict_tensorflow(file,title,tf_net)
    else:
        return await predict_yolov3(net,file,obj)

