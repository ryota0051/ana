from typing import Tuple

import numpy as np
import cv2


def get_average_of_circles(
        circles: np.ndarray,
        num_circles: int) -> Tuple[int, int, int]:
    '''ハフ変換で取得した円の中心座標と半径それぞれの平均値を返す関数

    Args:
        circles: ハフ変換で検出した円
        num_circles: 検出した円の数
    '''
    average_x = 0
    average_y = 0
    average_r = 0
    for i in range(num_circles):
        average_x = average_x + circles[0][i][0]
        average_y = average_y + circles[0][i][1]
        average_r = average_r + circles[0][i][2]
    average_x = int(average_x / num_circles)
    average_y = int(average_y / num_circles)
    average_r = int(average_r / num_circles)
    return average_x, average_y, average_r


def dist2pts(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calc_rad(x, y):
    return np.arctan(np.divide(float(y), float(x)))


def img2gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
