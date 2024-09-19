# Therapist / Child Detection and Tracking

![](videos/demo/Matching-ezgif.com-video-to-gif-converter.gif)

## Overview

The assignment aims to build a person detector (specifically, only, child and adult) along with a tracking approach, to assign unique IDs to persons, and track them throughout the video. 
  
The proposed method should be able to:
- Assign Unique IDs: The aim is to assign unique IDs to persons and track them throughout the video.
- Track Re-entries: The proposed method should be able to track the person if he/she goes out and re-enters the frame. This includes multiple children and adults.
- Assign New IDs: Assign a new ID to a person entering the frame for the first time.
- Post-Occlusion Tracking: Re-track the person and assign the correct ID within the video duration, post-occlusion, or partial visibility.

## Approach

### 1. Data Annotation with Roboflow

To develop the detection model, i first annotated the dataset using Roboflow. This involved the following steps:

- **Upload Images**: Upload a set of images into Roboflow.
- **Draw Bounding Boxes**: Used Roboflow's annotation tools to manually draw bounding boxes around the children and therapists in each image.
- **Label Objects**: Assigned labels to each bounding box to identify the person as either a "child" or "therapist".
- **Generate Dataset**: Exported the annotated dataset in a format suitable for training the [YOLO V8](https://github.com/ultralytics/ultralytics) object detection model

You can see the data set here [Children and Adults DataSet](https://universe.roboflow.com/a-4euhx/children-vs-adults-yolo-my3ct)

### 2. Model Training

- **Preprocess Data**: Converted and organized the dataset according to the requirements of the [YOLO V8](https://github.com/ultralytics/ultralytics) model.
- **Train Model**: Trained the model on the annotated dataset, fine-tuning hyperparameters to optimize performance.

### 3. Tracking Implementation

The tracking implementation is divided into two main components: the SORT model for initial tracking and a ReID model for handling identity re-identification.

#### 3.1. DeepSort Model
DeepSORT is a tracking algorithm which tracks object not only based on the velocity and motion of the object but also based on the appearance of the object.

- **DeepSORT Model**: Integrated the DeepSORT model to track adults and children across frames using bounding box data.
- **Track Re-entries**: Configured the max_age paremter of DeepSort to handle scenarios where individuals may leave and re-enter the frame, ensuring that their identities are consistently tracked.
- **Enhanced DeepSORT**: Modified the DeepSORT model to return a unique Object ID (OID) that includes the class id of the label (child or adult) for each tracked individual.

You can see [My DeepSort model](https://github.com/Dighzy/deep_sort_pytorch) or the [Original DeepSort Model](https://github.com/ZQPei/deep_sort_pytorch)

#### 3.2. FastReID Model
[FastReID](https://github.com/JDAI-CV/fast-reid) is a research platform that implements state-of-the-art re-identification algorithms.

- **Implement ReID Model**: Incorporated the Fast ReID model to enhance tracking performance by re-identifying individuals who may have been occluded or left the frame.
- **Manage Occlusions**: Developed algorithms to use ReID for accurately re-tracking individuals who were partially or fully occluded, ensuring consistent identity assignment throughout the video.

I used the BoT (Bag of Tricks and A Strong Baseline for Deep Person Re-identification pretrained models) pretrained model to perform my model.
You can see all the models [here](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md)

#### 3.3. BONUS: Emotion Recognition
To provide a deeper understanding of the interactions between the therapist and the child, I incorporated an emotion recognition feature into the tracking system.

- **Emotion Recognition**: Integrated an emotion recognition model to analyze facial expressions within the tracked bounding boxes. This allows for real-time detection and classification of emotions such as happiness, sadness, surprise, etc.
- **Implementation**: The [emotion recognition model](https://github.com/otaha178/Emotion-recognition) was applied to the faces detected within the bounding boxes of tracked individuals. The model processes these faces and outputs a predicted emotion label.

This bonus feature enhances the overall system by not only tracking identities but also providing emotional context, which can be crucial in understanding interactions in therapeutic settings.

### 4. Inference Pipeline

- **Video Processing**: Created a pipeline to process long-duration videos, overlaying bounding boxes and unique IDs on the footage.
- **Generate Output**: Produced output videos with predictions, including labeled bounding boxes and IDs.

### 5. Testing and Evaluation

- **Test Videos**: Applied the pipeline to test videos to evaluate its performance.
- **Analyze Results**: Assessed the accuracy of detections and tracking, making necessary adjustments.
- **Challenges and Future Work**: The model occasionally struggles with maintaining accuracy in challenging environments, such as when there are overlapping objects, low lighting, or rapid movements. These scenarios can cause confusion in tracking and emotion recognition. With further refinement, including additional training on a more diverse dataset and fine-tuning of the model, the system's robustness in such difficult conditions can be significantly improved.

## Demo videos
You can see all the best results in the best_results folder

![](videos/demo/result_3-ezgif.com-video-to-gif-converter.gif)
![](videos/demo/result_1-ezgif.com-video-to-gif-converter.gif)


## Dependencies

| **Core Libraries**         | **Additional Libraries**                       |
|----------------------------|------------------------------------------------|
| `Cython==3.0.11`           | `gdown==5.2.0`                                |
| `faiss==1.7.4`             | `imageio==2.35.1`                             |
| `h5py==3.11.0`             | `matplotlib==3.7.5`                           |
| `numpy==1.23.5`            | `opencv_python==4.10.0.84`                    |
| `scikit_learn==0.21.1`     | `Pillow==10.4.0`                              |
| `scipy==1.14.1`            | `pytubefix==6.14.0`                           |
| `torch==2.4.0+cu124`       | `PyYAML==6.0.2`                               |
| `torchvision==0.19.0+cu124` | `tabulate==0.9.0`                             |
| `ultralytics==8.2.83`      | `termcolor==2.4.0`                            |
| `imutils==0.5.3`            | `tqdm==4.66.5`                                |
|                            | `yacs==0.1.8`                                |



## Quick Start

1. **Install Dependencies**  

   Ensure all necessary dependencies are installed by running:
    ```bash
       pip install -r requirements.txt
    ```

2. **Download YouTube Videos**

    Use the utility script to download the required YouTube videos:
    ```bash
    python src/utils.py
    ```
3. **Update Paths**

    Modify the absolute path in the YAML configuration files:
    ````bash
     Children and Adults DataSet/data.yaml
    ````
4. **Retrain the Model (Optional)**
 
    If you wish to retrain the model, execute the following command:
    ```bash
   python src/train.py
    ```
5. **Test Detection on a Single Video**

   To test detection on an individual video, run:
    ```bash
    python src/test.py
    ```

6. Run Main Script for Full Detection

    Execute the main script to perform detection across all videos:
    ```bash
    python src/main.py
    ```

## References
- code: [zengwb-lx/Yolov5-Deepsort-Fastreid](https://github.com/zengwb-lx/Yolov5-Deepsort-Fastreid/tree/main)
- code: [nwojke/deep_sort](https://github.com/nwojke/deep_sort)
- paper: [Real Time Deep SORT](https://learnopencv.com/real-time-deep-sort-with-torchvision-detectors/)
- paper: [DeepSort : A Machine Learning Model for Tracking People](https://medium.com/axinc-ai/deepsort-a-machine-learning-model-for-tracking-people-1170743b5984)
- paper: [FastReID: A Pytorch Toolbox](https://arxiv.org/abs/2006.02631)
- paper: [Faster Person Re-Identification](https://arxiv.org/abs/2008.06826)
- code: [Gagan824/Computer-Vision-Projects](https://github.com/Gagan824/Computer-Vision-Projects/tree/main)
- video: [Real-Time Object Tracking using YOLOv8 and DeepSORT](https://www.youtube.com/watch?v=9jRRZ-WL698&list=PLoO1lozcNA1eTcvaf560hLwwLsKVwfdMb&index=9)
- paper [opencv-tutorial-face](https://opencv-tutorial.readthedocs.io/en/latest/face/face.html)
- code [Emotion-recognition](https://github.com/otaha178/Emotion-recognition)
- code [Facial Expression Recognition](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/rules)