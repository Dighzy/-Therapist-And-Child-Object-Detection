import cv2 as cv
import numpy as np

import torch

import src.utils as utils
import warnings
from pathlib import Path

from keras.utils.image_utils import img_to_array

utils.set_seed()
warnings.filterwarnings("ignore", category=DeprecationWarning)


def model_pipeline(path, yolo_model, tracker, face_detection, emotion_classifier, emotions, detection_threshold):

    #Open the video
    cap = cv.VideoCapture(path)
    path = Path(path)

    #Get the size of my original video
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    size = (frame_width, frame_height)

    # Create the VideoWriter object
    result = cv.VideoWriter(f'../videos/detection_and_tracking/{path.stem}.mp4', cv.VideoWriter_fourcc(*'mp4v') , 10, size)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # Getting my prediction
            results = yolo_model(frame)

            # Definning the variables
            bboxes_xywh = []
            confs = []
            class_id = []
            faces = []
            gray = []

            # Getting the boxes, classes and confidence
            boxes = results[0].boxes  # Convert tensor to list
            xyxy = boxes.xyxy
            conf = boxes.conf
            xywh = boxes.xywh

            cls = boxes.cls.tolist()
            class_dict = dict(yolo_model.names)

            # Append the values in a list only if the confidence is more or equal to the threshold
            for c, b, co in zip(cls, xywh, conf.cpu().numpy()):
                if co >= detection_threshold:
                    bboxes_xywh.append(b.cpu().numpy())
                    confs.append(co)
                    class_id.append(int(c))

            # If we have detections, use the sort method to get the tracks and outputs
            if bboxes_xywh:
                bboxes_xywh = torch.Tensor(np.array(bboxes_xywh))  # Convert to tensor
                confs = torch.Tensor(confs)  # Convert to tensor

                outputs, _ = tracker.update(bboxes_xywh, confs, class_dict, frame, oids=class_id)

                # Iterations in the tracker object
                for track in tracker.tracker.tracks:

                    alpha = 0.6
                    label_face = []

                    track_id = track.track_id
                    hits = track.hits
                    covariance = track.covariance
                    track_oid = track.oid

                    # Get bounding box coordinates in (x1, y1, x2, y2) format and the w and h
                    x1, y1, x2, y2 = track.to_tlbr()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w = x2 - x1  # Calculate width
                    h = y2 - y1  # Calculate height

                    # get the detection img to process the face
                    detection_img = frame[y1:y2, x1:x2, :]

                    # Process the faces
                    if detection_img.size > 0:
                        gray = cv.cvtColor(detection_img, cv.COLOR_BGR2GRAY)

                        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)

                        # If we have faces, process the emotions
                        if len(faces) > 0:
                            # Sort faces by size and take the largest one
                            faces = sorted(faces, reverse=True, key=lambda x: x[2] * x[3])[0]
                            (fX, fY, fW, fH) = faces


                            # Get the Region of Interest(roi) of my faces
                            roi = gray[fY:fY + fH, fX:fX + fW]
                            roi = cv.resize(roi, (64, 64))
                            roi = roi.astype("float") / 255.0
                            roi = img_to_array(roi)
                            roi = np.expand_dims(roi, axis=0)

                            # Get the predctions, probability and label
                            preds = emotion_classifier.predict(roi)[0]
                            emotion_probability = np.max(preds)
                            label = emotions[preds.argmax()]

                            # Adjust the face coordinates relative to the original frame
                            fX += x1
                            fY += y1

                            # Emotions label
                            label_face = '{:}-{:.2f}'.format(label, emotion_probability)

                            # Draw a semi-transparent (faded) rectangle for the face object
                            overlay = frame.copy()
                            cv.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (255, 0, 0), 1)
                            cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                    # Get the color based on the oid
                    color = classes_color(track_oid)

                    # Object label
                    label = 'Id: {:}-{:}'.format(track_id, class_dict.get(track_oid))
                    t_size_label = cv.getTextSize(label, cv.FONT_HERSHEY_PLAIN, 0.75, 1)[0]

                    # Draw a Rectangle for the detected object
                    cv.rectangle(frame, (x1, y1), (x2, y2), color, 1, lineType=cv.LINE_AA)

                    # draw a label into the inside top of the rectangle
                    # Draw a semi-transparent (faded) rectangle for the text on the original frame
                    overlay = frame.copy()
                    cv.rectangle(overlay, (x1, y1), (x1 + t_size_label[0] + 4, y1 + t_size_label[1] + 4), color, -1, lineType=cv.LINE_AA)
                    cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                    cv.putText(frame, label, (x1, y1 + t_size_label[1] + 4), cv.FONT_HERSHEY_PLAIN, 0.75, [0, 0, 0], 1)

                    # If label face draw a label into the inside bottom of the rectangle
                    if label_face:
                        t_size_face = cv.getTextSize(label_face, cv.FONT_HERSHEY_PLAIN, 0.75, 1)[0]

                        # Draw a semi-transparent (faded) rectangle for the text on the original frame
                        overlay = frame.copy()
                        cv.rectangle(overlay, (x1, y1 + h), (x1 + t_size_face[0] + 4, y1 + h - t_size_face[1] - 4), color, -1, lineType=cv.LINE_AA)
                        cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                        cv.putText(frame, label_face, (x1 + 4, y1 + h - t_size_face[1] + 4), cv.FONT_HERSHEY_PLAIN, 0.75, [0, 0, 0], 1)

            cv.imshow('frame', frame)
            result.write(frame)

            # Press E on keyboard to  exit
            if cv.waitKey(1) == ord('e'):
                break
                # if frame is read correctly ret is True
        else:
            print("Can't receive frame (stream end?). Exiting ...")
            break

    cap.release()
    result.release()
    cv.destroyAllWindows()


def classes_color(class_id):
    # Adult
    if class_id == 0:
        # yellow color
        color = (0, 255, 255)

    # Child
    elif class_id == 1:
        # Magenta color
        color = (255, 0, 255)

    else:
        color = (187, 0, 0)

    return tuple(color)
