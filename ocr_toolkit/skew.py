"""
倾斜校正模块

自动检测文档图像的倾斜角度并进行矫正。
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def detect_skew_angle(image: NDArray[np.uint8]) -> float:
    """检测图像的倾斜角度。

    Parameters
    ----------
    image : ndarray
        输入图像（BGR 或灰度）。

    Returns
    -------
    float
        检测到的倾斜角度（度）。正值为逆时针，负值为顺时针。
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 反转前景和背景
    gray = cv2.bitwise_not(gray)

    # 使用 Otsu 阈值二值化
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # 获取所有非零像素坐标
    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) == 0:
        return 0.0

    # 计算最小面积矩形
    angle = cv2.minAreaRect(coords)[-1]

    # 修正角度范围
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    return float(angle)


def correct_skew(
    image: NDArray[np.uint8],
    angle: float | None = None,
    border_mode: int = cv2.BORDER_REPLICATE,
) -> NDArray[np.uint8]:
    """校正图像倾斜。

    Parameters
    ----------
    image : ndarray
        输入图像。
    angle : float or None
        旋转角度。如果为 None，则自动检测。
    border_mode : int
        边界填充模式，默认复制边界像素。

    Returns
    -------
    ndarray
        校正后的图像。
    """
    if angle is None:
        angle = detect_skew_angle(image)

    if abs(angle) < 0.1:
        return image.copy()

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=border_mode,
    )
    return rotated
