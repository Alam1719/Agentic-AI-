import cv2
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def load_model(prototxt_path, model_path):
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return net

def draw_predictions(frame, detections, confidence_threshold=0.8):
    (h, w) = frame.shape[:2]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < confidence_threshold:
            continue

        class_id = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        (startX, startY, endX, endY) = box.astype("int")

        label = f"{CLASSES[class_id]}: {confidence * 100:.1f}%"

        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame