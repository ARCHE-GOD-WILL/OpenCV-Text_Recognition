"""
图像切片模块

根据检测到的线条坐标将表格图像切割为独立的单元格图像。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from ocr_toolkit.line_detect import DetectedLines


@dataclass
class CellImage:
    """切割出的单元格图像。

    Attributes
    ----------
    image : ndarray
        单元格图像数据。
    row : int
        行索引（从 0 开始）。
    col : int
        列索引（从 0 开始）。
    x : int
        左上角 X 坐标。
    y : int
        左上角 Y 坐标。
    w : int
        宽度。
    h : int
        高度。
    """

    image: NDArray[np.uint8]
    row: int
    col: int
    x: int
    y: int
    w: int
    h: int


def slice_by_lines(
    image: NDArray[np.uint8],
    y_positions: list[int],
    include_boundaries: bool = True,
) -> list[NDArray[np.uint8]]:
    """按水平线位置将图像切割为多个行条带。

    Parameters
    ----------
    image : ndarray
        输入图像。
    y_positions : list[int]
        水平线 Y 坐标列表。
    include_boundaries : bool
        是否包含图像顶部和底部作为边界。

    Returns
    -------
    list[ndarray]
        切割后的行图像列表。
    """
    h = image.shape[0]
    positions = sorted(y_positions)

    if include_boundaries:
        if not positions or positions[0] > 0:
            positions.insert(0, 0)
        if positions[-1] < h:
            positions.append(h)

    strips: list[NDArray[np.uint8]] = []
    for i in range(len(positions) - 1):
        y_start = positions[i]
        y_end = positions[i + 1]
        if y_end - y_start > 2:  # 跳过太窄的条带
            strips.append(image[y_start:y_end, :].copy())

    return strips


def slice_grid(
    image: NDArray[np.uint8],
    detected_lines: DetectedLines | None = None,
    y_positions: list[int] | None = None,
    x_positions: list[int] | None = None,
    include_boundaries: bool = True,
    min_cell_size: int = 5,
) -> list[CellImage]:
    """根据水平线和垂直线将图像切割为网格单元格。

    Parameters
    ----------
    image : ndarray
        输入图像。
    detected_lines : DetectedLines or None
        检测到的线条结果。如果提供，优先使用。
    y_positions : list[int] or None
        水平线 Y 坐标列表。
    x_positions : list[int] or None
        垂直线 X 坐标列表。
    include_boundaries : bool
        是否包含图像边界。
    min_cell_size : int
        最小单元格尺寸（像素）。

    Returns
    -------
    list[CellImage]
        切割后的单元格列表，按行优先排序。
    """
    h, w = image.shape[:2]

    if detected_lines is not None:
        y_pos = list(detected_lines.horizontal)
        x_pos = list(detected_lines.vertical)
    else:
        y_pos = list(y_positions) if y_positions else []
        x_pos = list(x_positions) if x_positions else []

    y_pos = sorted(y_pos)
    x_pos = sorted(x_pos)

    if include_boundaries:
        if not y_pos or y_pos[0] > 0:
            y_pos.insert(0, 0)
        if not y_pos or y_pos[-1] < h:
            y_pos.append(h)
        if not x_pos or x_pos[0] > 0:
            x_pos.insert(0, 0)
        if not x_pos or x_pos[-1] < w:
            x_pos.append(w)

    # 如果没有垂直线，整行作为一个单元格
    if len(x_pos) < 2:
        x_pos = [0, w]

    cells: list[CellImage] = []
    for row_idx in range(len(y_pos) - 1):
        y_start = y_pos[row_idx]
        y_end = y_pos[row_idx + 1]
        if y_end - y_start < min_cell_size:
            continue

        for col_idx in range(len(x_pos) - 1):
            x_start = x_pos[col_idx]
            x_end = x_pos[col_idx + 1]
            if x_end - x_start < min_cell_size:
                continue

            cell_img = image[y_start:y_end, x_start:x_end].copy()
            cells.append(
                CellImage(
                    image=cell_img,
                    row=row_idx,
                    col=col_idx,
                    x=x_start,
                    y=y_start,
                    w=x_end - x_start,
                    h=y_end - y_start,
                )
            )

    return cells


def save_cells(
    cells: list[CellImage],
    output_dir: str,
    prefix: str = "cell",
    ext: str = ".jpg",
) -> list[str]:
    """将单元格图像保存到指定目录。

    Parameters
    ----------
    cells : list[CellImage]
        单元格列表。
    output_dir : str
        输出目录路径。
    prefix : str
        文件名前缀。
    ext : str
        文件扩展名。

    Returns
    -------
    list[str]
        保存的文件路径列表。
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    paths: list[str] = []
    for cell in cells:
        filename = f"{prefix}_{cell.row}_{cell.col}{ext}"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, cell.image)
        paths.append(filepath)

    return paths
