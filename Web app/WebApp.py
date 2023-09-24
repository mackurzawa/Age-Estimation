import os
import sys

from Model import SimpleModel, create_resnet, create_densenet
from Predict import predict
from Preprocess import find_face
import torch
import torchvision.models as models
import torch.nn as nn


import cv2
import streamlit as st
import numpy as np
import time


def crop_predict_paint(frame, face, model):
    x, y, w, h = face
    xxx = 15
    cv2.rectangle(frame, (x - xxx, y - xxx), (x + w + xxx, y + h + xxx), (0, 255, 0), 4)

    face = frame[y+xxx:y+h+xxx, x+xxx:x+w+xxx]
    pred = predict(face, model)
    pred_class = np.argmax(pred)

    del last_ages[0]
    last_ages.append(CLASS_NAMES[pred_class])

    dominant_class = max(set(last_ages), key=last_ages.count)

    cv2.putText(frame, dominant_class + 'Prob: {:.2f}'.format(max(pred)), (x, y), font,
               fontScale, color, thickness, cv2.LINE_AA)


font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2

CLASS_NAMES = ['1-15', '16-30', '31-45', '46-60', '61-116']
model = SimpleModel(len(CLASS_NAMES))
# model = models.resnet50(pretrained=True)
# model.fc = nn.Sequential(
#     nn.Linear(2048, 512),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(512, 128),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(128, len(CLASS_NAMES))
# )
model_name = os.listdir(os.path.join('Structurized Files', 'Models', 'Simple'))[-1]
MODEL_PATH = os.path.join('Structurized Files', 'Models', 'Simple', model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))
print(f'Model {model_name} loaded')

st.title("Webcam Live Age Prediction")
run = True
FRAME_WINDOW = st.image([])
st.title('Drop here your mp4!')
st.text('App will detect the face and predict the age of the person in the video!\n')
uploaded_video = st.file_uploader('Movie upload', accept_multiple_files=False, type=['mp4'])
VIDEO_WINDOW = st.empty()
camera = cv2.VideoCapture(0)
time.sleep(2)
video_uploaded = False
LAST_AGES_LIMIT = 10
last_ages = [CLASS_NAMES[0] for _ in range(LAST_AGES_LIMIT)]

if not camera.isOpened():
    print('Camera is not working :(')
    sys.exit()

while run:
    _, frame = camera.read()
    if frame is None:
        st.title('Refresh Page! There was a problem with your webcam!')
        sys.exit()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = find_face(frame)

    if face is not None:
        crop_predict_paint(frame, face, model)
    FRAME_WINDOW.image(frame)

    if uploaded_video:
        with open('temp.mp4', 'wb') as f:
            f.write(uploaded_video.read())
        vidcap = cv2.VideoCapture('temp.mp4')
        frame_width = int(vidcap.get(3))
        frame_height = int(vidcap.get(4))
        size = (frame_width, frame_height)
        result = cv2.VideoWriter('temp_written.mp4', cv2.VideoWriter_fourcc(*'H264'), 30, size)

        frame_count = 0
        while True:
            frame_count += 1
            success, frame = vidcap.read()  # get next frame from video
            if not success:
                break

            face = find_face(frame)
            if face is not None:
                crop_predict_paint(frame, face, model)
            result.write(frame)
            dots = frame_count//10 % 3 + 1
            VIDEO_WINDOW.header('Working ' + "".join(['.' for _ in range(dots)]))
        vidcap.release()
        result.release()
        time.sleep(0.5)
        video_uploaded = True
        VIDEO_WINDOW.video(open('temp_written.mp4', 'rb').read())
        os.remove('temp.mp4')
        os.remove('temp_written.mp4')
        st.balloons()
        uploaded_video = None
