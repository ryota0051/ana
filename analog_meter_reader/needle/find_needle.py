from typing import Tuple, Union

import cv2
import numpy as np

from analog_meter_reader.utils import dist2pts


def get_needle_cood(
        gray: np.ndarray,
        center_x: int,
        center_y: int,
        r: int) -> Union[Tuple[int, int, int, int], None]:
    '''
    '''
    _, gray = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)
    lines = cv2.HoughLinesP(
        image=gray,
        rho=3,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=10,
        maxLineGap=0
    )
    diff1_lower_bound = 0.15
    diff1_upper_bound = 0.5
    diff2_lower_bound = 0.5
    diff2_upper_bound = 1.0
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            diff1 = dist2pts(center_x, center_y, x1, y1)
            diff2 = dist2pts(center_x, center_y, x2, y2)
            if (diff1 > diff2):
                diff1, diff2 = diff2, diff1
            if (
                    (diff1 < diff1_upper_bound * r) and
                    (diff1 > diff1_lower_bound * r) and
                    (diff2 < diff2_upper_bound * r) and
                    (diff2 > diff2_lower_bound * r)
            ):
                return x1, y1, x2, y2
    return None
