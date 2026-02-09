"""
图像预处理模块

提供颜色去除、锐化、降噪、直方图均衡化等图像预处理功能。
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def remove_red_color(
    image: NDArray[np.uint8],
    lower_hue_range: tuple[int, int] = (0, 10),
    upper_hue_range: tuple[int, int] = (170, 180),
    saturation_min: int = 50,
    value_min: int = 50,
) -> NDArray[np.uint8]:
    """移除图像中的红色（如印章、红色线条等），将其替换为白色。

    Parameters
    ----------
    image : ndarray
        BGR 格式输入图像。
    lower_hue_range : tuple[int, int]
        HSV 色调低范围 (min, max)，默认 (0, 10)。
    upper_hue_range : tuple[int, int]
        HSV 色调高范围 (min, max)，默认 (170, 180)。
    saturation_min : int
        最低饱和度阈值，默认 50。
    value_min : int
        最低明度阈值，默认 50。

    Returns
    -------
    ndarray
        红色区域被替换为白色后的图像。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 低色调范围红色掩码
    lower1 = np.array([lower_hue_range[0], saturation_min, value_min])
    upper1 = np.array([lower_hue_range[1], 255, 255])
    mask_low = cv2.inRange(hsv, lower1, upper1)

    # 高色调范围红色掩码
    lower2 = np.array([upper_hue_range[0], saturation_min, value_min])
    upper2 = np.array([upper_hue_range[1], 255, 255])
    mask_high = cv2.inRange(hsv, lower2, upper2)

    mask = mask_low | mask_high

    result = image.copy()
    result[mask != 0] = 255
    return result


def sharpen_image(
    image: NDArray[np.uint8],
    kernel_size: int = 40,
    strength: float = 2.0,
) -> NDArray[np.uint8]:
    """使用自定义高通滤波器锐化图像。

    Parameters
    ----------
    image : ndarray
        输入图像（灰度或彩色）。
    kernel_size : int
        卷积核大小，默认 40。
    strength : float
        锐化强度，默认 2.0。

    Returns
    -------
    ndarray
        锐化后的图像。
    """
    kernel = np.zeros((kernel_size, kernel_size), np.float32)
    center = kernel_size // 2
    kernel[center, center] = strength

    box_filter = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    kernel = kernel - box_filter

    return cv2.filter2D(image, -1, kernel)


def denoise_image(
    image: NDArray[np.uint8],
    h: int = 10,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> NDArray[np.uint8]:
    """使用非局部均值降噪。

    Parameters
    ----------
    image : ndarray
        输入图像。
    h : int
        滤波强度，默认 10。较大的值去除更多噪点但也会丢失细节。
    template_window_size : int
        模板块大小（奇数），默认 7。
    search_window_size : int
        搜索窗口大小（奇数），默认 21。

    Returns
    -------
    ndarray
        降噪后的图像。
    """
    if len(image.shape) == 2 or image.shape[2] == 1:
        return cv2.fastNlMeansDenoising(image, None, h, template_window_size, search_window_size)
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, template_window_size, search_window_size)


def enhance_image(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """通过直方图均衡化增强图像对比度。

    Parameters
    ----------
    image : ndarray
        输入图像（灰度或 BGR）。

    Returns
    -------
    ndarray
        增强后的图像。
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    return cv2.equalizeHist(gray)


def preprocess_image(
    image: NDArray[np.uint8],
    remove_red: bool = True,
    sharpen: bool = True,
    denoise: bool = True,
    enhance: bool = False,
    sharpen_kernel_size: int = 40,
    denoise_strength: int = 10,
) -> NDArray[np.uint8]:
    """一站式图像预处理流水线。

    Parameters
    ----------
    image : ndarray
        BGR 格式输入图像。
    remove_red : bool
        是否移除红色。
    sharpen : bool
        是否锐化。
    denoise : bool
        是否降噪。
    enhance : bool
        是否直方图均衡化（会转为灰度）。
    sharpen_kernel_size : int
        锐化核大小。
    denoise_strength : int
        降噪强度。

    Returns
    -------
    ndarray
        预处理后的图像。
    """
    result = image.copy()

    if remove_red:
        result = remove_red_color(result)

    if sharpen:
        result = sharpen_image(result, kernel_size=sharpen_kernel_size)

    if denoise:
        result = denoise_image(result, h=denoise_strength)

    if enhance:
        result = enhance_image(result)

    return result
