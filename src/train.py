from ultralytics import YOLO
import utils


def train_model(model_yolo, data, epochs, patience):
    model = YOLO(model_yolo)
    model.train(data=data, epochs=epochs, patience=patience)

    return model


def model_val(model):
    metrics = model.val()
    print(metrics)


if __name__ == '__main__':
    utils.set_seed()

    model = "../models/yoloModels/yolov8m.pt"
    data = "../Children and Adults DataSet/data.yaml"  # change the absolute path into this file
    epochs = 100
    patience = 25

    model = train_model(model, data, epochs, patience)

    model_val(model)
