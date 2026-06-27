import os
import cv2

try:
    from src.utils import load_model, draw_predictions
except ImportError:
    from utils import load_model, draw_predictions

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROTOTXT = os.path.join(BASE_DIR, "models", "MobileNetSSD_deploy.prototxt")
CAFFEMODEL = os.path.join(BASE_DIR, "models", "MobileNetSSD_deploy.caffemodel")

net = load_model(PROTOTXT, CAFFEMODEL)


def detect_objects(image, confidence_threshold=0.8):
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), (127.5, 127.5, 127.5))
    net.setInput(blob)
    detections = net.forward()
    output_image = draw_predictions(image, detections, confidence_threshold)
    return output_image


if __name__ == "__main__":
    IMAGE_PATH = os.path.join(BASE_DIR, "sample_images", "download.webp")
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("ERROR: Could not load image. Check the file path:", IMAGE_PATH)
        exit()

    result = detect_objects(image)
    output_path = os.path.join(BASE_DIR, "outputs", "result.jpg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"Done! Check {output_path}")