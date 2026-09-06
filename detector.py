from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class PotholeDetector:
    """
    YOLO-first pothole detector.

    Expected YOLO model:
      models/best.pt

    The model should contain a pothole class. If a YOLO model is not
    available, a simple OpenCV fallback is used for demonstration.
    """

    def __init__(self, model_path="models/best.pt", confidence=0.35, iou=0.45):
        self.confidence = confidence
        self.iou = iou
        self.model = None

        if YOLO is not None and Path(model_path).exists():
            self.model = YOLO(model_path)

    def predict(self, frame):
        if self.model is not None:
            return self._yolo_predict(frame)
        return self._opencv_fallback(frame)

    def _yolo_predict(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            verbose=False
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy.tolist()
                conf = float(box.conf[0].cpu().item())

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                })

        return detections

    def _opencv_fallback(self, frame):
        """
        Demonstration-only fallback.

        It searches for dark/irregular regions. This is NOT a trained
        pothole classifier and should not be used for safety-critical
        road assessment.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)

        threshold = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51, 9
        )

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h, w = frame.shape[:2]
        frame_area = h * w
        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < frame_area * 0.002:
                continue
            if area > frame_area * 0.20:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)

            if bw < 30 or bh < 20:
                continue

            # Approximate confidence based on region characteristics.  
            rectangularity = area / max(bw * bh, 1)
            conf = min(0.90, max(0.20, rectangularity))

            if conf >= self.confidence:
                detections.append({
                    "bbox": [x, y, x+bw, y+bh],
                    "confidence": float(conf),
                })

        return detections
