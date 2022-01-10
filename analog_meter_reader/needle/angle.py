from typing import Tuple

import cv2
import numpy as np

from analog_meter_reader.utils import dist2pts, calc_rad

REFERENCE_START_ANGLE = 35
REFERENCE_END_ANGLE = 330


def get_needle_angle(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        center_x: int,
        center_y: int) -> float:
    dist_pt_0 = dist2pts(center_x, center_y, x1, y1)
    dist_pt_1 = dist2pts(center_x, center_y, x2, y2)
    if (dist_pt_0 > dist_pt_1):
        x_for_angle = x1 - center_x
        y_for_angle = center_y - y1
    else:
        x_for_angle = x2 - center_x
        y_for_angle = center_y - y2

    rad = calc_rad(x_for_angle, y_for_angle)
    deg = np.rad2deg(rad)
    if x_for_angle > 0 and y_for_angle > 0:
        result = 270 - deg
    if x_for_angle < 0 and y_for_angle > 0:
        result = 90 - deg
    if x_for_angle < 0 and y_for_angle < 0:
        result = 90 - deg
    if x_for_angle > 0 and y_for_angle < 0:
        result = 270 - deg
    return float(result)


def get_needle_angle_range(
        gray: np.ndarray,
        center_x: int,
        center_y: int,
        r: int,
        interval: int = 36,
        separation: int = 10) -> Tuple[float, float]:
    # 必要な領域抽出
    def get_meter_region(img, vertices):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img
    vertices = np.zeros((interval, 2))
    for i in range(interval):
        angle = separation * i * np.pi / 180
        for j in range(2):
            vertices[i][j] = center_x + 0.8 * r * np.cos(angle) if j % 2 == 0 \
                else center_y + 0.8 * r * np.sin(angle)
    # 輪郭抽出
    canny = cv2.Canny(gray, 200, 20)
    cropped_img = get_meter_region(canny, np.array([vertices], np.int32))
    contours, _ = cv2.findContours(
        cropped_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    target_cnt_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 15:
            target_cnt_list.append(contour)
    # 角度の最小値と最大値取得
    min_angle = 90
    max_angle = 270
    for i in range(len(target_cnt_list)):
        cnt = target_cnt_list[i]
        cnt = cnt.reshape(len(cnt), 2)
        x1, y1 = np.mean(cnt, axis=0)
        x_len = x1 - center_x
        y_len = center_y - y1
        rad = calc_rad(abs(x_len), abs(y_len))
        deg = np.rad2deg(rad)
        if x_len < 0 and y_len < 0:
            start_angle = 90 - deg
            if start_angle > REFERENCE_START_ANGLE:
                if start_angle < min_angle:
                    min_angle = start_angle
        elif x_len > 0 and y_len < 0:
            end_angle = 270 + deg
            if end_angle < REFERENCE_END_ANGLE:
                if end_angle > max_angle:
                    max_angle = end_angle
    return float(min_angle), float(max_angle)


def angle2meter_value(
        needle_angle: float,
        min_angle,
        max_angle,
        min_value,
        max_value) -> float:
    value_range = max_value - min_value
    angle_range = max_angle - min_angle
    angle_ratio = (needle_angle - min_angle) / angle_range
    return float(angle_ratio * value_range + min_value)
