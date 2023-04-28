import cv2


color = (0, 255, 0)
thickness = 2


def overlap_area(x1, y1, w1, h1, x2, y2, w2, h2):
    """
    Calculates the area of overlap between two rectangles.
    rect1 and rect2 should be tuples of (x, y, width, height),
    representing the top-left coordinates and dimensions of each rectangle.
    """

    # Calculate the coordinates of the overlapping rectangle
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    # Calculate the area of the overlapping rectangle
    overlap_area = x_overlap * y_overlap

    return overlap_area

# results_yolov8 : yolov8x-seg.pt로 predict 호출 후 리턴값
# want_title : 예를 들어 bus면 bus가 포함 된 것만 선택
def select_16x16(results_yolov8,image_ndarray,want_title,cv2_test_mode=False):
    selected = []
    y = image_ndarray.shape[0]  # 전체 이미지 기준
    x = image_ndarray.shape[1]
    w_quarter = x // 4
    h_quarter = y // 4
    one_part_area = w_quarter * h_quarter  # 16개 중에서 하나의 넓이

    threshold_area = 0.5 # ex) 16개 부분 중에서 한 부분에서, 박스가 포함되는 부분의 넓이가 한 부분의 넓이의 50% 이상
    class_numbers = list(map(int, results_yolov8[0].boxes.cls.numpy()))
    name_dictionary = results_yolov8[0].names
    class_names = list(name_dictionary[index] for index in class_numbers)
    for cur_index,name in enumerate(class_names):
        if name != want_title:
            continue
        box_x,box_y,box_w,box_h = map(int,list(results_yolov8[0].boxes.xyxy[cur_index].numpy()))
        if cv2_test_mode: # 해당 객체 전체
            cv2.rectangle(image_ndarray, (int(box_x), int(box_y)), (int(box_w), int(box_h)),
                          (0,0,255), thickness)
            cv2.imshow("선택", image_ndarray)
            cv2.waitKey(0)

        part_yx = [[0,0],[0,w_quarter],[0,(w_quarter)*2],[0,(w_quarter)*3]
            ,[h_quarter,0],[h_quarter,w_quarter],[h_quarter,(w_quarter)*2],[h_quarter,(w_quarter)*3]
            ,[(h_quarter)*2,0],[(h_quarter)*2,w_quarter],[(h_quarter)*2,(w_quarter)*2],[(h_quarter)*2,(w_quarter)*3]
            ,[(h_quarter)*3,0],[(h_quarter)*3,w_quarter],[(h_quarter)*3,(w_quarter)*2],[(h_quarter)*3,(w_quarter)*3]]
        for img_idx in range(len(part_yx)):
            y_cur = part_yx[img_idx][0]
            x_cur = part_yx[img_idx][1]
            o_area = overlap_area(x_cur,y_cur,w_quarter,h_quarter,box_x,box_y,box_w-box_x,box_h-box_y)
            if o_area != 0 and one_part_area//o_area > threshold_area:
                selected.append(img_idx)
                if cv2_test_mode:
                    cv2.rectangle(image_ndarray, (int(x_cur), int(y_cur)), (int(x_cur+w_quarter), int(y_cur+h_quarter)), color, thickness)

    if cv2_test_mode:
        cv2.imshow("선택",image_ndarray)
        cv2.waitKey(0)
    return selected


def rectangle_image_show(results_yolov8,origin_image_path):
    font = cv2.FONT_HERSHEY_PLAIN

    img = cv2.imread(origin_image_path)
    class_names = list(map(int, results_yolov8[0].boxes.cls.numpy()))
    # 박스들
    for idx, box in enumerate(results_yolov8[0].boxes):
        x, y, w, h = list(box.xyxy.numpy()[0])
        label_number = class_names[idx]  # ex) 0  (0이면 사람, 1이면 ~~)
        name = results_yolov8[0].names[label_number]  # ex) person
        img = cv2.putText(img, name, (int(x), int(y)), font, 2, (0, 0, 255), 1, cv2.LINE_AA)
        print(f"상자 {idx + 1} : {name}")
        cv2.rectangle(img, (int(x), int(y)), (int(w), int(h)), color, thickness)

    cv2.imshow('test_yolov8_rectangle', img)
    cv2.waitKey(0)
