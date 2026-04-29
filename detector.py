from ultralytics import YOLO
from config import YOLO_MODEL, DRONE_CLASSES, CONFIDENCE, IOU_THRESHOLD
class DroneDetector:

    def __init__(self):
        print(f" Завантаження моделі {YOLO_MODEL}...")
        self.model = YOLO(YOLO_MODEL)

    def detect(self, frame):
        """
        Args:
            frame: numpy array (BGR)

        Returns:
            list: [{'bbox': [x1,y1,x2,y2], 'conf': float, 'class': int}, ...]
        """
        results = self.model(
            frame,
            classes=DRONE_CLASSES,
            conf=CONFIDENCE,
            iou=IOU_THRESHOLD,
            verbose=False
        )

        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'conf': conf,
                'class': cls
            })

        return detections