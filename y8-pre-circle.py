from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO(r'runs\segment\train8\weights\best.pt')
model.export(format='onnx', imgsz=(544, 640))

# Run inference on 'bus.jpg' with arguments
model.predict(r'C:\Users\PC\Desktop\Freelancer\Aligh-with-arm\SyntheticData2\20240604_010524OriginalImage.jpg', save=True, imgsz=640, conf=0.1)