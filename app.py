import cv2
import numpy as np
from paddleocr import PaddleOCR
import gradio as gr
import matplotlib.pyplot as plt
import requests


# Predict function
def predict(input_image):

    model_cfg_path = 'darknet_yolo.cfg'
    model_weights_path = 'model.weights'
    model = cv2.dnn.readNetFromDarknet(
        cfgFile=model_cfg_path, darknetModel=model_weights_path)

    cv2.imwrite('0.jpg', input_image)

    img = plt.imread('0.jpg')
    height = img.shape[0]
    width = img.shape[1]

    blob_img = cv2.dnn.blobFromImage(
        image=img, scalefactor=1/255, size=(416, 416), mean=0, swapRB=True)
    model.setInput(blob_img)

    layers = model.getLayerNames()
    output_layers = [layers[i - 1] for i in model.getUnconnectedOutLayers()]
    all_detections = model.forward(output_layers)
    detections = [
        detection for detections in all_detections for detection in detections if detection[4] * 10000 > 0.33]

    bboxes = []
    class_ids = []
    scores = []

    for detection in detections:
        bbox = detection[:4]
        x_center, y_center, w, h = bbox
        bbox = [int(x_center * width), int(y_center * height),
                int(w * width), int(h * height)]
        bbox_confidence = detection[4]
        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)

    if all(item == 0 for item in scores):
        scores[0] = 0.4

    indices = cv2.dnn.NMSBoxes(
        bboxes=bboxes, scores=scores, score_threshold=0.33, nms_threshold=0.5)
    x_center, y_center, w, h = bboxes[indices[0]]

    print(bboxes)


    license_plate = img[int(y_center - (h / 2)):int(y_center + (h / 2)+10),
                        int(x_center - (w / 2)):int(x_center + (w / 2)+10), :].copy()
    license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    _, license_plate_thresh = cv2.threshold(
        license_plate_gray, 150, 255, cv2.THRESH_BINARY)
    _, license_plate_thresh_2 = cv2.threshold(
        license_plate_gray, 64, 255, cv2.THRESH_BINARY)

    final_output = {}

    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr.ocr(license_plate_thresh, det=False, cls=False)
    if result[0][0][0] == '':
        result = ocr.ocr(license_plate_thresh_2, det=False, cls=False)
    for i in range(len(result)):
        result = result[i]
        final_output = {"".join(
            e for e in line[0] if e.isalnum()): f'{line[-1]:.2f}' for line in result}

    license_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB)
    x1 = int(x_center - (w / 2))
    y1 = int(y_center - (h / 2))
    x2 = int(x_center + (w / 2)+10)
    y2 = int(y_center + (h / 2)+10) 
    rectangle_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rectangle_img = cv2.rectangle(rectangle_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("License Plate", rectangle_img)
    cv2.waitKey(0)
    return license_plate, final_output


url = 'https://img.autoabc.lv/BMW-X5/BMW-X5_2018_Apvidus_21111351946_11.jpg'
response = requests.get(url)
input_image = cv2.imdecode(np.frombuffer(
    response.content, np.uint8), cv2.IMREAD_COLOR)

license_plate, final_output = predict(input_image)
cv2.imshow("License Plate", license_plate)
cv2.waitKey(0)

print('Result:', final_output)
