import cv2
import numpy as np
import onnxruntime as ort
import time
from picamera2 import Picamera2

CLASS_NAMES = ["person"]
CONFIDENCE_THRESHOLD = 0.4
NMS_IOU_THRESHOLD = 0.45
MODEL_INPUT_SIZE = (640, 640)  # width, height

def setup_camera(width=640, height=480):
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    return picam2

def preprocess(image, input_size):
    img = cv2.resize(image, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB888 from PiCam to BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Then to RGB as model expects
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC  CHW
    img = np.expand_dims(img, axis=0) 
    return img

def postprocess(predictions, input_size, original_size):
    boxes, scores, class_ids = [], [], []

    for pred in predictions:
        x, y, w, h = pred[0:4]
        obj_conf = pred[4]
        cls_conf = pred[5:].max()
        cls_id = pred[5:].argmax()
        score = obj_conf * cls_conf

        if score > CONFIDENCE_THRESHOLD:
            scale_x = original_size[0] / input_size[0]
            scale_y = original_size[1] / input_size[1]

            x *= scale_x
            y *= scale_y
            w *= scale_x
            h *= scale_y

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            boxes.append([x1, y1, int(w), int(h)])
            scores.append(float(score))
            class_ids.append(int(cls_id))

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD)

    results = []
    for i in indices:
        idx = int(i)
        x, y, w, h = boxes[idx]
        results.append((x, y, x + w, y + h, scores[idx], class_ids[idx]))
    return results

def draw_boxes(frame, detections):
    for (x1, y1, x2, y2, conf, cls_id) in detections:
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class {cls_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def main():
    picam2 = setup_camera()
    print("Camera ready")

    model_path = "best_pruned_quantized.onnx" 
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"Loaded model: {model_path}")

    try:
        while True:
            frame = picam2.capture_array()  # RGB888 format
            input_size = MODEL_INPUT_SIZE
            original_size = (frame.shape[1], frame.shape[0])  # width, height

            input_tensor = preprocess(frame, input_size)
            output = session.run(None, {input_name: input_tensor})[0]  # shape: [1, N, 85]
            predictions = output[0] 

            detections = postprocess(predictions, input_size, original_size)
            annotated = draw_boxes(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), detections)

            cv2.imshow("YOLO Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print("Error during detection:", e)
    finally:
        cv2.destroyAllWindows()
        picam2.stop()
        print("Camera stopped")

if __name__ == "__main__":
    main()
