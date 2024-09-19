from ultralytics import YOLO
from deep_sort_pytorch.deep_sort.deep_sort import DeepSort

from keras.models import load_model
from model import model_pipeline

import cv2 as cv
from pathlib import Path


if __name__ == '__main__':

    # Load an official or custom Yolo model
    # model = YOLO ('yoloModels/yolov8m.pt')
    yolo_model = YOLO("../runs/detect/train4/weights/best.pt")

    # Load the reid models
    reid_weights = '../models/reid_models/market_bot_R50.pth'
    reid_model_config = '../models/reid_models/bagtricks_R50.yml'

    # Load the DeepSort model and select its parameters
    tracker = DeepSort(model_path=reid_weights, model_config=reid_model_config, max_age=60, n_init=5, max_iou_distance=0.7, nn_budget=500)

    # Load the emotion recognition models
    detection_model_path = '../models/emotion_recognition_models/haarcascade_frontalface_default.xml'
    emotion_model_path = '../models/emotion_recognition_models/_mini_XCEPTION.102-0.66.hdf5'
    face_detection = cv.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)

    # Array of the emotions
    emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

    #Select the detection threshold
    detection_threshold = 0.60

    folder_path = Path('../videos/originals/')

    # Iteration in all videos in the folder
    for videos in folder_path.iterdir():
        videos_paths = videos

        model_pipeline(videos_paths, yolo_model, tracker, face_detection, emotion_classifier, emotions, detection_threshold)

