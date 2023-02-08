from flask import Flask, request
import yolov5
import cv2
from paddleocr import PaddleOCR
import numpy as np
import requests

app = Flask(__name__)

# load model
model = yolov5.load('keremberke/yolov5n-license-plate')

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per boxcls
model.max_det = 1000  # maximum number of detections per image

# OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

@app.route('/predict', methods=['GET'])
def predict():
    # get image URL from request
    img_url = request.args.get('img_url')

    # download image
    response = requests.get(img_url)
    input_image = cv2.imdecode(np.frombuffer(
        response.content, np.uint8), cv2.IMREAD_COLOR)

    # perform inference
    results = model(input_image, size=640)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4]  # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    # convert boxes to rectangles
    for box in boxes:
        x1, y1, x2, y2 = box
        h, w = y2 - y1, x2 - x1
        x_center, y_center = x1 + (w / 2), y1 + (h / 2)

        license_plate = input_image[int(y_center - (h / 2)):int(y_center + (h / 2)),
                                    int(x_center - (w / 2)):int(x_center + (w / 2)), :].copy()
        license_plate_gray = cv2.cvtColor(
            license_plate, cv2.COLOR_BGR2GRAY)
        _, license_plate_thresh = cv2.threshold(
            license_plate_gray, 150, 255, cv2.THRESH_BINARY)
        _, license_plate_thresh_2 = cv2.threshold(
            license_plate_gray, 64, 255, cv2.THRESH_BINARY)

        result = ocr.ocr(license_plate_thresh, det=False, cls=False)

        return result

if __name__ == '__main__':
    app.run(debug=True)