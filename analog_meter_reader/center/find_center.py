from typing import Tuple, Union
import numpy as np
import cv2

from analog_meter_reader.utils import get_average_of_circles


def get_circle_center_and_radius(
        gray: np.ndarray) -> Union[Tuple[int, int, int], None]:
    '''対象画像から、メータの中心座標と半径を取得する

    Args:
        img: 入力画像

    Returns:
        (メータの中心x座標, メータの中心y座標, メータの半径)
    '''
    height, _ = gray.shape[:2]
    circles: Union[None, np.ndarray] = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        circles=np.array([]),
        param1=100,
        param2=50,
        minRadius=int(height * 0.35),
        maxRadius=int(height * 0.5)
    )
    if circles is None:
        return None
    return get_average_of_circles(circles, circles.shape[1])
