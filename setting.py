import configparser
import os.path


"""-
< 구글 데모 >
pageurl = 'https://www.google.com/recaptcha/api2/demo'
sitekey = '6Le-wvkSAAAAAPBMRTvw0Q4Muexq9bi0DJwx_mJ-'
"""

def make_config():
    config = configparser.ConfigParser()
    config['solve_page'] = {}
    config['solve_page']['pageurl'] = 'https://www.google.com/recaptcha/api2/demo' # 구글 데모 사이트
    config['solve_page']['sitekey'] = '6Le-wvkSAAAAAPBMRTvw0Q4Muexq9bi0DJwx_mJ-' # 구글 데모 사이트
    with open('config.ini', 'w', encoding='utf-8') as configfile:
        config.write(configfile)

def get_config():
    # config 없으면 생성
    if not os.path.exists('config.ini'):
        make_config()

    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    pageurl = config['solve_page']['pageurl']
    sitekey = config['solve_page']['sitekey']
    return pageurl,sitekey


pageurl,sitekey = get_config()

base_link = f"C:\\Users\\vvpas\\PycharmProjects\\recaptcha_solver\\model"
use_tf_model_label = {'Bus','Car','Crosswalk','Hydrant','Mountain','Palm','Traffic Light'} # yolov3 대신 텐서플로우 모델로 예측할 레이블
tf_labels = ['Bus','Car','Crosswalk','Hydrant','Mountain','Palm','Traffic Light'] # 학습된 모델 레이블 순서
tf_model_path = f'/tensorflow_model/saved.h5'  # 텐서플로우 모델
tf_optimizer = 'adam'
tf_loss_function = 'sparse_categorical_crossentropy'
# yolo v3 모델 정보
yolov3_txt_path = base_link+'\\yolov3.txt'
yolov3_weights = base_link+'\\yolov3.weights'
yolov3_cfg = base_link+'\\yolov3.cfg'