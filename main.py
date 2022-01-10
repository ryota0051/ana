import argparse
import os

import cv2
import numpy as np

from analog_meter_reader.center.find_center import (
    get_circle_center_and_radius
)
from analog_meter_reader.needle.find_needle import get_needle_cood
from analog_meter_reader.utils import img2gray
from analog_meter_reader.needle.angle import (
    get_needle_angle_range,
    get_needle_angle,
    angle2meter_value
)


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('min', type=float, help='メータの最小値')
    parser.add_argument('max', type=float, help='メータの最大値')
    parser.add_argument('--src', help='入力画像名', default='meter3.png')
    parser.add_argument('--unit', help='メータの単位', default='℃')
    return parser.parse_args()


def main(img: np.ndarray, min_value, max_value):
    gray = img2gray(img)
    # メータの中心取得
    circle_info = get_circle_center_and_radius(gray)
    if circle_info is None:
        print('画像にメータが見つかりませんでした')
        exit(-1)
    center_x, center_y, r = circle_info
    # メータの針の角度範囲取得
    separation = 10
    interval = int(360 / separation)
    min_angle, max_angle = get_needle_angle_range(
        gray,
        center_x,
        center_y,
        r,
        interval,
        separation
    )
    x1, y1, x2, y2 = get_needle_cood(gray, center_x, center_y, r)
    angle = get_needle_angle(
        x1,
        y1,
        x2,
        y2,
        center_x,
        center_y
    )
    result = angle2meter_value(
        angle,
        min_angle,
        max_angle,
        min_value,
        max_value
    )
    return result


if __name__ == '__main__':
    arg = get_arg()
    src = f'./images/{arg.src}'
    if not os.path.exists(src):
        print('ファイルが存在しません')
        exit(-1)
    img = cv2.imread(src)
    result = main(img, min_value=arg.min, max_value=arg.max)
    print(f'{result} {arg.unit}')
