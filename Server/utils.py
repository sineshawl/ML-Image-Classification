import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import toWavelet


def classify_image(base_64_data, file_path=None):
    print(base_64_data)
def get_base64_for_teddy():
    with open('base64.txt') as file:
        return file.read()
    
def get_cv2_image_from_base64_string(b64str):

    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_image_with_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('E:\\RoadMap\\ML Projects\\ML-Image-Classification\\opencv\\haarcascade\\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('E:\\RoadMap\\ML Projects\\ML-Image-Classification\\opencv\\haarcascade\\haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)
    return cropped_faces

def get_b64_test_image_for_virat():
    with open("b64.txt") as f:
        return f.read()

if __name__ == '__main__':
    load_saved_artifacts()

    print(classify_image(get_b64_test_image_for_virat(), None))

if '__name__' == '__main__':
    print(classify_image(get_base64_for_teddy(), None))