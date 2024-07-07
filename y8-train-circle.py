from ultralytics import YOLO



model = YOLO(r'runs\segment\train7\weights\best.pt')


results = model.train(data='align-circle.yaml', epochs=30, imgsz=640, batch=16, translate=0, scale=0, flipud=0.5, crop_fraction=0.1, mosaic=0, val=False)
