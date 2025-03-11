import re
from collections import defaultdict


def is_russian_license_plate(plate):
    """
    Проверяет, является ли строка российским номером автомобиля.

    Формат: 1 буква, 3 цифры, 2 буквы, 2-3 цифры (регион).
    Разрешённые буквы: А, В, Е, К, М, Н, О, Р, С, Т, У, Х.
    """
    # Регулярное выражение для проверки базового формата
    pattern = r'^[АВЕКМНОРСТУХ]{1}\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$'

    # Проверка формата с помощью регулярного выражения
    if not re.match(pattern, plate):
        return False

    # Извлечение региона
    try:
        # Последние 2 или 3 символа должны быть цифрами (код региона)
        region = plate[-3:] if plate[-3:].isdigit() else plate[-2:]
        region = int(region)  # Попытка преобразования в число
    except ValueError:
        # Если регион не является числом, это не РФ номер
        return False

    # Проверка диапазона региона (1–999)
    if 1 <= region <= 999:
        return True

    return False


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1


def calculate_most_common_plate(license_plate_texts):
    """
    Находит наиболее вероятный номер и его уверенность.

    :param license_plate_texts: Список кортежей в формате (номер, уверенность)
    :return: Кортеж (номер, уверенность)
    """
    # Создаем словарь для хранения уверенности по номерам
    data = defaultdict(list)
    for plate, confidence in license_plate_texts:
        data[plate].append(confidence)

    # Рассчитываем среднюю уверенность и взвешенный скоринг
    max_score = 0
    most_probable_plate = None

    for plate, confidences in data.items():
        avg_confidence = sum(confidences) / len(confidences)  # Средняя уверенность
        score = avg_confidence * len(confidences)  # Взвешенный скоринг
        if score > max_score:
            max_score = score
            most_probable_plate = plate

    # Вернем наиболее вероятный номер и его уверенность (среднюю)
    return most_probable_plate, max_score / len(data[most_probable_plate]) if most_probable_plate else 0

