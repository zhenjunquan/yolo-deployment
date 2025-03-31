import argparse
import time
import cv2
import numpy as np
import yaml


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, colors, classes):
    label = f"{classes[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def process_frame(frame, model, colors, classes, score_threshold, nms_threshold):
    height, width, _ = frame.shape
    length = max(height, width)
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = frame
    scale = length / 640

    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)
    start = time.time()
    outputs = model.forward()
    end = time.time()
    elapsed_time = end - start

    # 计算帧率
    fps = 1.0 / elapsed_time

    # 打印帧率
    print("[INFO] YOLO FPS: {:.2f}".format(fps))
    outputs = np.array([cv2.transpose(outputs[0])])

    boxes, scores, class_ids = [], [], []
    for i in range(outputs.shape[1]):
        classes_scores = outputs[0][i][4:]
        _, maxScore, _, maxClassIndex = cv2.minMaxLoc(classes_scores)
        if maxScore >= score_threshold:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(int(maxClassIndex[0]))

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold, 0.5)
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        draw_bounding_box(
            frame, class_ids[index], scores[index],
            round(box[0] * scale), round(box[1] * scale),
            round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale),
            colors, classes
        )
    return frame


def main(onnx_model, yaml_file, video_source, score_threshold, nms_threshold):
    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        classes = data["names"]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    model = cv2.dnn.readNetFromONNX(onnx_model)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    video_source = int(video_source) if video_source.isdigit() else video_source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, model, colors, classes, score_threshold, nms_threshold)
        cv2.imshow("Video Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="onnx_model/yolo11n.onnx", help="Path to ONNX model.")
    parser.add_argument("--yaml", default="config/coco.yaml", help="Path to YAML file containing class names.")
    parser.add_argument("--source", default='0', help="Video source (0 for webcam, path for video file).")
    parser.add_argument("--score_threshold", type=float, default=0.25, help="Confidence score threshold.")
    parser.add_argument("--nms_threshold", type=float, default=0.45, help="Non-maximum suppression threshold.")
    args = parser.parse_args()

    main(args.model, args.yaml, args.source, args.score_threshold, args.nms_threshold)
