"""
线条检测模块

使用 Canny 边缘检测 + Hough 变换检测图像中的水平线和垂直线。
主要用于表格结构识别。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class DetectedLines:
    """检测到的线条结果。

    Attributes
    ----------
    horizontal : list[int]
        水平线的 Y 坐标列表（已排序、去重）。
    vertical : list[int]
        垂直线的 X 坐标列表（已排序、去重）。
    image_height : int
        原始图像高度。
    image_width : int
        原始图像宽度。
    """

    horizontal: list[int]
    vertical: list[int]
    image_height: int
    image_width: int


def _extract_line_segments(
    edges: NDArray[np.uint8],
    hough_threshold: int = 150,
    min_line_length: int = 120,
    max_line_gap: int = 5,
) -> list[list[int]]:
    """从边缘图提取线段。

    Returns
    -------
    list[list[int]]
        每个元素为 [x1, y1, x2, y2]。
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        return []

    return [[int(l[0][0]), int(l[0][1]), int(l[0][2]), int(l[0][3])] for l in lines]


def _deduplicate_positions(positions: list[int], min_gap: int = 40) -> list[int]:
    """去重位置坐标：合并距离小于 min_gap 的坐标。"""
    if not positions:
        return []

    positions = sorted(positions)
    result = [positions[0]]
    for pos in positions[1:]:
        if abs(pos - result[-1]) > min_gap:
            result.append(pos)
    return result


def detect_horizontal_lines(
    image: NDArray[np.uint8],
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 150,
    min_line_length: int = 120,
    max_line_gap: int = 5,
    angle_tolerance: int = 10,
    min_length: int = 10,
    min_gap: int = 40,
) -> list[int]:
    """检测图像中的水平线。

    Parameters
    ----------
    image : ndarray
        输入图像。
    canny_low : int
        Canny 低阈值。
    canny_high : int
        Canny 高阈值。
    hough_threshold : int
        Hough 变换投票阈值。
    min_line_length : int
        最小线段长度。
    max_line_gap : int
        最大间隔允许值。
    angle_tolerance : int
        水平线 Y 偏差容差（像素）。
    min_length : int
        水平线最小 X 跨度。
    min_gap : int
        去重最小间距。

    Returns
    -------
    list[int]
        水平线 Y 坐标列表（排序后、去重后）。
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
    segments = _extract_line_segments(edges, hough_threshold, min_line_length, max_line_gap)

    # 按 Y 坐标排序
    segments.sort(key=lambda s: s[1])

    y_positions: list[int] = []
    for x1, y1, x2, y2 in segments:
        if abs(y1 - y2) < angle_tolerance and abs(x1 - x2) > min_length:
            y_positions.append(y1)

    return _deduplicate_positions(y_positions, min_gap)


def detect_vertical_lines(
    image: NDArray[np.uint8],
    canny_low: int = 20,
    canny_high: int = 40,
    hough_threshold: int = 10,
    min_line_length: int = 0,
    max_line_gap: int = 0,
    angle_tolerance: int = 10,
    min_height: int = 5,
    min_gap: int = 40,
) -> list[int]:
    """检测图像中的垂直线。

    Parameters
    ----------
    image : ndarray
        输入图像。
    canny_low : int
        Canny 低阈值。
    canny_high : int
        Canny 高阈值。
    hough_threshold : int
        Hough 变换投票阈值。
    min_line_length : int
        最小线段长度。
    max_line_gap : int
        最大间隔允许值。
    angle_tolerance : int
        垂直线 X 偏差容差（像素）。
    min_height : int
        垂直线最小 Y 跨度。
    min_gap : int
        去重最小间距。

    Returns
    -------
    list[int]
        垂直线 X 坐标列表（排序后、去重后）。
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
    segments = _extract_line_segments(edges, hough_threshold, min_line_length, max_line_gap)

    # 按 X 坐标排序
    segments.sort(key=lambda s: s[0])

    x_positions: list[int] = []
    for x1, y1, x2, y2 in segments:
        if abs(y1 - y2) > min_height and abs(x1 - x2) < angle_tolerance:
            x_positions.append(x1)

    return _deduplicate_positions(x_positions, min_gap)


def detect_all_lines(
    image: NDArray[np.uint8],
    h_canny_low: int = 50,
    h_canny_high: int = 150,
    h_hough_threshold: int = 150,
    h_min_line_length: int = 120,
    v_canny_low: int = 20,
    v_canny_high: int = 40,
    v_hough_threshold: int = 10,
    min_gap: int = 40,
) -> DetectedLines:
    """同时检测水平线和垂直线。

    Parameters
    ----------
    image : ndarray
        输入图像。
    h_canny_low, h_canny_high : int
        水平线 Canny 阈值。
    h_hough_threshold : int
        水平线 Hough 阈值。
    h_min_line_length : int
        水平线最小长度。
    v_canny_low, v_canny_high : int
        垂直线 Canny 阈值。
    v_hough_threshold : int
        垂直线 Hough 阈值。
    min_gap : int
        去重最小间距。

    Returns
    -------
    DetectedLines
        包含水平线和垂直线坐标的结果对象。
    """
    h, w = image.shape[:2]

    horizontal = detect_horizontal_lines(
        image,
        canny_low=h_canny_low,
        canny_high=h_canny_high,
        hough_threshold=h_hough_threshold,
        min_line_length=h_min_line_length,
        min_gap=min_gap,
    )

    vertical = detect_vertical_lines(
        image,
        canny_low=v_canny_low,
        canny_high=v_canny_high,
        hough_threshold=v_hough_threshold,
        min_gap=min_gap,
    )

    return DetectedLines(
        horizontal=horizontal,
        vertical=vertical,
        image_height=h,
        image_width=w,
    )
