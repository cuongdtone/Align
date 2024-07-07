from ultralytics import YOLO

# Load a model
model = YOLO(r'runs\detect\train19\weights\best.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data=r'align.yaml', 
                      epochs=30, 
                      imgsz=920, 
                      batch=16,
                      hsv_s=0.2,
                      workers=4,
                      device=['cpu'],
                      flipud=0, 
                      fliplr=0,
                      val=True,
                      mosaic=0
)
