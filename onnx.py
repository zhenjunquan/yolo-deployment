import argparse
import cv2
import numpy as np
import onnxruntime as ort
import yaml
import time


class YOLOv8:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, yaml_file, input_source, confidence_thres, iou_thres):
        """
        Initializes an instance of the YOLOv8 class.
        """
        self.input_source = int(input_source) if input_source.isdigit() else input_source
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load COCO classes
        with open(yaml_file, "r", encoding="utf-8") as f:
            self.classes = yaml.safe_load(f)["names"]

        # Generate color palette
        self.color_palette = np.random.randint(0, 255, size=(len(self.classes), 3), dtype=np.uint8)

        # ONNX Runtime session with optimization
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_model, sess_options=session_options,
                                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # Get input shape
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape
        self.input_width, self.input_height = self.input_shape[2], self.input_shape[3]

    def draw_detections(self, img, box, score, class_id):
        """Draws bounding boxes and labels."""
        x1, y1, w, h = box
        color = self.color_palette[class_id].tolist()
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Draw rectangle and label background
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, 2)
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1, label_height + 5)
        cv2.rectangle(img, (x1, label_y - label_height), (x1 + label_width, label_y), color, cv2.FILLED)

        # Put label text
        cv2.putText(img, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, img):
        """Preprocess the input image."""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # Channel first
        return np.expand_dims(img, axis=0)

    def postprocess(self, input_image, output):
        """Extracts bounding boxes, scores, and class IDs."""
        outputs = np.squeeze(output[0]).T
        img_h, img_w = input_image.shape[:2]
        scale_w, scale_h = img_w / self.input_width, img_h / self.input_height

        # Extract scores, boxes, and class ids
        scores = np.max(outputs[:, 4:], axis=1)
        valid_indices = np.where(scores >= self.confidence_thres)[0]
        class_ids = np.argmax(outputs[valid_indices, 4:], axis=1)
        boxes = outputs[valid_indices, :4]

        # Convert to absolute coordinates
        boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * scale_w
        boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * scale_h
        boxes[:, 2] *= scale_w
        boxes[:, 3] *= scale_h
        boxes = boxes.astype(int)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores[valid_indices].tolist(),
                                   self.confidence_thres, self.iou_thres)

        if len(indices) > 0:
            for i in indices.flatten():
                self.draw_detections(input_image, boxes[i], scores[valid_indices[i]], class_ids[i])

        return input_image

    def process_frame(self, frame):
        """Processes a single frame."""
        img_data = self.preprocess(frame)

        start_time = time.time()
        output = self.session.run(None, {self.input_name: img_data})
        fps = 1.0 / (time.time() - start_time)
        print(f"[INFO] YOLO FPS: {fps:.2f}")

        return self.postprocess(frame.copy(), output)

    def run_video_detection(self):
        """Runs object detection on a video stream."""
        cap = cv2.VideoCapture(self.input_source)

        if not cap.isOpened():
            raise RuntimeError("无法打开视频源")

        cv2.namedWindow("YOLOv8 Real-time Detection", cv2.WINDOW_NORMAL)

        # Pre-warm the camera
        for _ in range(5):
            cap.read()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow("YOLOv8 Real-time Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="onnx_model/yolo11n.onnx", help="Path to ONNX model.")
    parser.add_argument("--yaml", default="config/coco.yaml", help="Path to YAML file containing class names.")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam or video file path).")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    args = parser.parse_args()

    detector = YOLOv8(args.model, args.yaml, args.source, args.conf_thres, args.iou_thres)
    detector.run_video_detection()
