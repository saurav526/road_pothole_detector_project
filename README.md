# 🚧 Smart Road Pothole Detection System

Interactive pothole detection using **OpenCV + YOLO + Streamlit**.

## Features

- Image pothole detection
- Video frame-by-frame detection
- Adjustable confidence and IoU thresholds
- Bounding boxes
- Confidence scores
- Low / Moderate / Severe classification
- Detection statistics
- Confidence analytics
- Processed video download
- OpenCV fallback when a trained YOLO model is not present

## 1. Create environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

## 2. Install packages

```bash
pip install -r requirements.txt
```

## 3. Add the trained model

Put your pothole YOLO model here:

```text
models/best.pt
```

The model should have a pothole class.

## 4. Run

```bash
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

## 5. Training your own model

Use a YOLO-format pothole dataset.

Expected structure:

```text
data/
├── pothole.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Example `pothole.yaml`:

```yaml
path: ./data
train: images/train
val: images/val

names:
  0: pothole
```

Then:

```bash
python train.py
```

Copy:

```text
runs/detect/pothole_detector/weights/best.pt
```

to:

```text
models/best.pt
```

## Important

The OpenCV fallback is only a demonstration. It detects dark/irregular regions and can produce false positives from shadows, cracks, road patches, and other objects.

For a real project, use a properly trained pothole YOLO model and evaluate it on a held-out road dataset.
