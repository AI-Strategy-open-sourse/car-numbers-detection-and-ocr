import argparse
import os

import numpy as np
from easyocr import easyocr
from ultralytics import YOLO
import cv2

from sort import Sort

from util import get_car, is_russian_license_plate, calculate_most_common_plate


def read_license_plate(license_plate_image, params):
    reader = easyocr.Reader(
        ['ru'],
        gpu=True,
    )

    result = reader.readtext(
        license_plate_image,
        beamWidth=10,
        batch_size=4,
        allowlist='АВЕКМНОРСТУХ0123456789',
        contrast_ths=params.get('contrast_ths', 0.2),  # Значение по умолчанию — 0.2
        low_text=params.get('low_text', 0.5),
        link_threshold=params.get('link_threshold', 0.4),
        canvas_size=params.get('canvas_size', 800),
        mag_ratio=params.get('mag_ratio', 1.5),
        text_threshold=params.get('text_threshold', 0.6),
        min_size=params.get('min_size', 5),
        paragraph=False,
        rotation_info=None,
        detail=1,
        bbox_min_score=0.5,
        bbox_min_size=5,
        max_candidates=0,
        output_format='standard'
    )

    combined_number = ''
    confidences = []

    for each in result:
        text = each[1]  # Распознанный текст
        confidence = each[2]  # Уверенность
        combined_number += text
        confidences.append(confidence)

    combined_number = combined_number.replace(' ', '')

    if confidences:
        average_confidence = sum(confidences) / len(confidences)
    else:
        average_confidence = 0.0

    return combined_number, average_confidence


def main(params):
    results = {}
    license_plate_texts = []

    mot_tracker = Sort()

    # load models
    coco_model = YOLO('yolov8n.pt', verbose=False)
    license_plate_detector = YOLO('license_plate_detector.pt', verbose=False)

    # load video
    cap = cv2.VideoCapture(0)

    vehicles = [2, 3, 5, 7]

    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}

            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

            try:
                track_ids = mot_tracker.update(np.asarray(detections_))
            except Exception as e:
                print(e)
                continue

            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255,
                                                                 cv2.THRESH_BINARY_INV)

                    license_plate_text, confidence = read_license_plate(license_plate_crop, params=params)

                    # print(f'DEBUG {license_plate_text} - {score=}')

                    if license_plate_text is not None and is_russian_license_plate(license_plate_text):
                        license_plate_texts.append((license_plate_text, confidence))

                    try:
                        most_common_plate, probability = calculate_most_common_plate(license_plate_texts)
                        if most_common_plate:
                            print(f"С наибольшей вероятностью ({probability:.2f}) номер автомобиля: {most_common_plate}")
                    except Exception as e:
                        print(f"Ошибка при анализе номеров: {e}")
                        print(f"Данные: {license_plate_texts}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--location_type',
        choices=['street', 'inner'],
        required=True,
        help='Тип местоположения: "street" для улицы или "inner" для внутренней стороны.'
    )
    args = parser.parse_args()

    if args.location_type == 'street':
        print("Анализируем уличную сторону.")
        main(params={'contrast_ths': 0.1, 'low_text': 0.6, 'link_threshold': 0.5, 'canvas_size': 1000, 'mag_ratio': 1.0, 'text_threshold': 0.4, 'min_size': 10})
    elif args.location_type == 'inner':
        print("Анализируем внутреннюю сторону.")
        main(params={'contrast_ths': 0.1, 'low_text': 0.6, 'link_threshold': 0.3, 'canvas_size': 1000, 'mag_ratio': 2.0, 'text_threshold': 0.4, 'min_size': 10})


