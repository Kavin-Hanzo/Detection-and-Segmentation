import cv2
import torch
import numpy as np
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
from torchvision.models.segmentation import deeplabv3_resnet50
from urllib.parse import unquote


class AIVisionProcessor:
    def __init__(self, camera_source=0, mode="object_detection"):
        """Initialize object detection and semantic segmentation models."""
        self.camera = cv2.VideoCapture(camera_source)
        self.current_mode = mode

        # Object Detection Model (YOLO)
        self.object_detection_model = YOLO('yolov8s.pt')

        # Semantic Segmentation Model (DeepLabV3)
        self.segmentation_model = deeplabv3_resnet50(pretrained=True)
        self.segmentation_model.eval()

        # COCO Class Names for object detection
        self.coco_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # Fixed colors for segmentation classes
        self.fixed_colors = [
            (0, 0, 0),        # 0: Background
            (0, 0, 255),      # 1: Aeroplane (Red)
            (0, 255, 0),      # 2: Bicycle (Green)
            (255, 0, 0),      # 3: Bird (Blue)
            (0, 255, 255),    # 4: Boat (Cyan)
            (255, 255, 0),    # 5: Bottle (Yellow)
            (255, 165, 0),    # 6: Bus (Orange)
            (0, 255, 255),    # 7: Car (Magenta)
            (128, 0, 128),    # 8: Cat (Purple)
            (255, 192, 203),  # 9: Chair (Pink)
            (128, 128, 0),    # 10: Cow (Olive)
            (255, 215, 0),    # 11: Dining Table (Gold)
            (255, 105, 180),  # 12: Dog (HotPink)
            (0, 128, 128),    # 13: Horse (Teal)
            (255, 69, 0),     # 14: Motorbike (Red-Orange)
            (255, 0, 255),    # 15: Person (Magenta)
            (144, 238, 144),  # 16: Potted Plant (LightGreen)
            (255, 20, 147),   # 17: Sheep (DeepPink)
            (0, 255, 127),    # 18: Sofa (SpringGreen)
            (255, 140, 0),    # 19: Train (DarkOrange)
            (0, 0, 128),      # 20: TV (Navy)
        ]

    def detect_objects(self, frame):
        """Perform object detection using YOLO."""
        results = self.object_detection_model.predict(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                if 0 <= cls < len(self.coco_names):
                    label = f"{self.coco_names[cls]} {conf:.2f}"
                else:
                    label = f"Unknown {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def semantic_segmentation(self, frame):
        """Perform semantic segmentation with fixed colors for each class."""
        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (513, 513))

        input_tensor = torch.from_numpy(input_image).permute(2, 0, 1).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = self.segmentation_model(input_tensor)['out']

        mask = output.squeeze().argmax(0).numpy()
        mask_resized = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Apply fixed colors for each class
        color_mask = np.zeros_like(frame, dtype=np.uint8)
        for class_id in range(len(self.fixed_colors)):
            color_mask[mask_resized == class_id] = self.fixed_colors[class_id]

        blended = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
        return blended

    def get_frame(self):
        """Capture and process video frame."""
        ret, frame = self.camera.read()
        if not ret:
            return None

        if self.current_mode == "object_detection":
            processed_frame = self.detect_objects(frame)
        elif self.current_mode == "semantic_segmentation":
            processed_frame = self.semantic_segmentation(frame)
        else:
            processed_frame = frame

        return processed_frame

    def change_camera(self, camera_source):
        """Change camera input dynamically."""
        self.camera.release()
        camera_source = unquote(camera_source)
        if camera_source.startswith("http"):
            self.camera = cv2.VideoCapture(camera_source)
        else:
            self.camera = cv2.VideoCapture(int(camera_source))

    def change_mode(self, mode):
        """Change the processing mode dynamically."""
        self.current_mode = mode


def create_app():
    app = Flask(__name__)
    vision_processor = AIVisionProcessor()

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/video_feed')
    def video_feed():
        def generate():
            while True:
                frame = vision_processor.get_frame()
                if frame is None:
                    break
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/change_camera/<camera_source>')
    def change_camera(camera_source):
        if camera_source.startswith('http'):
            camera_source = unquote(camera_source)
        vision_processor.change_camera(camera_source)
        return jsonify({"status": "success", "new_camera_source": camera_source})

    @app.route('/change_mode/<mode>')
    def change_mode(mode):
        vision_processor.change_mode(mode)
        return jsonify({"status": "success", "new_mode": mode})

    return app
