# import os
import cv2
import dlib
import numpy as np
# from zipfile import ZipFile
# from sklearn.preprocessing import OneHotEncoder

import utils

# import scikitplot
# import seaborn as sns
# from matplotlib import pyplot

# from sklearn.metrics import classification_report

from tensorflow import keras
# from keras import optimizers
# from keras.models import Model
# from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, Concatenate
# from keras.layers import Dropout, BatchNormalization
# from keras.utils import plot_model
# from keras.preprocessing.image import ImageDataGenerator
# import tensorflow as tf
# import math
# from collections import OrderedDict
# from imutils.face_utils import FaceAligner


import joblib
# import tensorflow as tf
# from keras import optimizers
# from sklearn.model_selection import train_test_split
# from keras.utils import plot_model
#
# from keras.callbacks import EarlyStopping
# from keras.callbacks import ReduceLROnPlateau
import time

from keras.models import load_model

HE = None
detector = 'dnn'
# input = "/Users/mohan/PycharmProjects/Bi-LSTM-CNN_FER/woman2.mp4"
# '/Users/mohan/PycharmProjects/DeepFace/upload/testvdo.mp4'
model_name = 'CNNModel_feraligned+ck_5emo'
model_name1 = 'CNNModel'
window_size = None

if HE is None:
    hist_eq = False
else:
    hist_eq = utils.arg2bool(HE)

# DLIB HoG
hog_detector = dlib.get_frontal_face_detector()

# Opencv DNN
modelFile = "./upload/dnn_tf.pb"
configFile = "./upload/dnn_tf.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
conf_threshold = 0.7


def dlib_detector(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    offset = 15
    x_pos, y_pos = 10, 40

    faces = hog_detector(gray_frame)
    for idx, face in enumerate(faces):
        if hist_eq:
            gray_frame = cv2.equalizeHist(gray_frame)

        img_arr = utils.align_face(gray_frame, face, desiredLeftEye)
        img_arr = utils.preprocess_img(img_arr, resize=False)

        predicted_proba = model.predict(img_arr)
        predicted_label = np.argmax(predicted_proba[0])

        x, y, w, h = utils.bb_to_rect(face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        text = f"Person {idx + 1}: {label2text[predicted_label]}"
        utils.draw_text_with_backgroud(frame, text, x + 5, y, font_scale=0.4)

        text = f"Person {idx + 1} :  "
        y_pos = y_pos + 2 * offset
        utils.draw_text_with_backgroud(frame, text, x_pos, y_pos, font_scale=0.3, box_coords_2=(2, -2))
        for k, v in label2text.items():
            text = f"{v}: {round(predicted_proba[0][k] * 100, 3)}%"
            y_pos = y_pos + offset
            utils.draw_text_with_backgroud(frame, text, x_pos, y_pos, font_scale=0.3, box_coords_2=(2, -2))


def dnn_detector(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    idx = 0
    offset = 15
    x_pos, y_pos = 10, 40

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            idx += 1
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            bboxes.append([x1, y1, x2, y2])
            # print(x1)
            # print(y1)

            face = [x1, y1, x2 - x1, y2 - y1]

            if hist_eq:
                gray_frame = cv2.equalizeHist(gray_frame)

            img_arr = utils.align_face(gray_frame, utils.bb_to_rect(face), desiredLeftEye)
            img_arr = utils.preprocess_img(img_arr, resize=False)

            predicted_proba = model.predict(img_arr)
            predicted_label = np.argmax(predicted_proba[0])



            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = f"Person {idx}: {label2text[predicted_label]}"
            utils.draw_text_with_backgroud(frame, text, x1 + 5, y1, font_scale=0.4)

            text = f"Person {idx} :  "
            y_pos = y_pos + 2 * offset
            utils.draw_text_with_backgroud(frame, text, x_pos, y_pos, font_scale=0.3, box_coords_2=(2, -2))
            for k, v in label2text.items():
                text = f"{v}: {round(predicted_proba[0][k] * 100, 3)}%"
                y_pos = y_pos + offset
                utils.draw_text_with_backgroud(frame, text, x_pos, y_pos, font_scale=0.3, box_coords_2=(2, -2))

            return label2text[predicted_label]


desiredLeftEye = (0.32, 0.32)
model = load_model("./upload/" + model_name + ".h5")
label2text = joblib.load("./upload/label2text_" + model_name + ".pkl")
emotion=""

'''
# For image prediction
def ImagePrediction(image_path, output_path, modelWeights):
    image_path = "./ski.jpg"
    output = "./videos/outputs/t.png"
    keras_weights_file = "model1.h5"

    tic = time.time()
    print('start processing...')
    model = Model.load_weights(keras_weights_file)

    # load config
    params, model_params = model.get_config()

    input_image = cv2.imread(image_path)  # B,G,R order
    body_parts, all_peaks, subset, candidate = cv2.extractChannel(input_image, params, model, model_params)
    canvas = cv2.drawMarker(input_image, all_peaks, subset, candidate)

    toc = time.time()
    print('processing time is %.5f' % (toc - tic))

    cv2.imwrite(output, canvas)

    cv2.destroyAllWindows()
'''
def store_emotion(predicted):
    global emotion
    emotion = predicted
    return emotion

def show_emotion() :
    global emotion
    return emotion


def detection_main():
    if input == "webcam":
        iswebcam = True
        vidcap = cv2.VideoCapture(0)
    else:
        iswebcam = False
        vidcap = cv2.VideoCapture(0)

    if not window_size is None:
        cv2.namedWindow("Facial Expression Recognition System", cv2.WINDOW_NORMAL)
        win_size = list(map(int, window_size.split(",")))
        cv2.resizeWindow("Facial Expression Recognition System", *win_size)

    frame_count = 0
    tt = 0
    n = 30
    while n:
        status, frame = vidcap.read()
        if not status:
            break

        frame_count += 1

        if iswebcam:
            frame = cv2.flip(frame, 1, 0)
        n-=1
        try:
            tik = time.time()

            if detector == "dlib":
                out = dlib_detector(frame)
            else:
                out = dnn_detector(frame)

            tt += time.time() - tik
            fps = frame_count / tt
            label = f"Detector: {detector} ; Model: {model_name1}"
            utils.draw_text_with_backgroud(frame, 10, 20, font_scale=0.35)

        except Exception as e:
            print(e)
            pass
        cv2.imshow("Facial Expression Recognition System", frame)
        if cv2.waitKey(10) == ord('q'):
            cv2.destroyAllWindows()
            vidcap.release()
            break
    emt = store_emotion(out)
    print(emt)







