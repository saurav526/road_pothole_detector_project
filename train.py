"""
Train a pothole detector.

1. Prepare a YOLO-format dataset:
   data/
     pothole.yaml
     images/train/
     images/val/
     labels/train/
     labels/val/

2. Example pothole.yaml:

   path: ./data
   train: images/train
   val: images/val

   names:
     0: pothole

3. Install ultralytics and run:
   python train.py

The resulting model will normally be under:
runs/detect/pothole_detector/weights/best.pt

Copy best.pt to:
models/best.pt

"""

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="data/pothole.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="runs/detect",
    name="pothole_detector",
)

print("Training finished. Copy the best.pt file to models/best.pt")
