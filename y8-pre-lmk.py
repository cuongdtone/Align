from ultralytics import YOLO
import cv2
from glob import glob

# Load a model
model = YOLO(r'runs\detect\train\weights\best.pt')  # load an official model
model.export(format='onnx')

# for i in glob(r'C:\Users\PC\Desktop\Freelancer\Aligh-with-arm\SyntheticData/*'):
for i in glob(r'align-lmk/*.jpg'):
    # Predict with the model

    results = model(i)  # predict on an image

    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        result.show()  # display to screen
        # result.save(filename='result.jpg')  # save to disk
    