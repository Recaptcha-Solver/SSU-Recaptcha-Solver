
kor_dict = {"버스":"bus","소화전":"fire hydrant",'오토바이':'motorcycles','신호등':'traffic light','자동차':'car'
            ,'자전거':'bicycle','횡단보도':'crosswalk'}


def convert_to_english(title):
    global kor_dict
    if title in kor_dict:
        return kor_dict[title]
    else:
        print(f"존재 하지 않는 키 : {title}")
        return title
