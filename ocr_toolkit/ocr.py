"""
OCR 文字识别模块

基于 Tesseract OCR 引擎，提供图像文字识别功能。
支持中文（简体/繁体）和英文识别。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytesseract
from numpy.typing import NDArray

from ocr_toolkit.slicer import CellImage


def _prepare_for_ocr(
    image: NDArray[np.uint8],
    threshold: bool = True,
    invert: bool = False,
) -> NDArray[np.uint8]:
    """为 OCR 准备图像：灰度化 + 可选二值化。"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if threshold:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    if invert:
        gray = cv2.bitwise_not(gray)

    return gray


def ocr_image(
    image: NDArray[np.uint8],
    lang: str = "chi_sim+eng",
    config: str = "--psm 6",
    threshold: bool = True,
    invert: bool = False,
) -> str:
    """对单张图像执行 OCR 文字识别。

    Parameters
    ----------
    image : ndarray
        输入图像。
    lang : str
        Tesseract 语言包，默认中英混合。
        常用值: "eng"（英文）、"chi_sim"（简体中文）、"chi_tra"（繁体中文）、
                "chi_sim+eng"（中英混合）。
    config : str
        Tesseract 配置参数。
        PSM 模式说明:
        - --psm 3: 全自动页面分割
        - --psm 6: 假设为统一文本块
        - --psm 7: 单行文本
        - --psm 8: 单词
        - --psm 13: 原始行
    threshold : bool
        是否自动二值化。
    invert : bool
        是否反转颜色（白字黑底 -> 黑字白底）。

    Returns
    -------
    str
        识别出的文本。
    """
    prepared = _prepare_for_ocr(image, threshold=threshold, invert=invert)
    text = pytesseract.image_to_string(prepared, lang=lang, config=config)
    return text.strip()


def ocr_cells(
    cells: list[CellImage],
    lang: str = "chi_sim+eng",
    config: str = "--psm 6",
    threshold: bool = True,
) -> dict[tuple[int, int], str]:
    """对多个单元格执行 OCR，返回 (行, 列) -> 文本 的映射。

    Parameters
    ----------
    cells : list[CellImage]
        单元格图像列表。
    lang : str
        Tesseract 语言包。
    config : str
        Tesseract 配置参数。
    threshold : bool
        是否自动二值化。

    Returns
    -------
    dict[tuple[int, int], str]
        字典，键为 (row, col) 元组，值为识别文本。
    """
    results: dict[tuple[int, int], str] = {}
    for cell in cells:
        text = ocr_image(cell.image, lang=lang, config=config, threshold=threshold)
        results[(cell.row, cell.col)] = text
    return results


def ocr_image_to_dataframe(
    cells: list[CellImage],
    lang: str = "chi_sim+eng",
    config: str = "--psm 6",
    threshold: bool = True,
):
    """对单元格执行 OCR 并返回 pandas DataFrame。

    Parameters
    ----------
    cells : list[CellImage]
        单元格图像列表。
    lang : str
        Tesseract 语言包。
    config : str
        Tesseract 配置参数。
    threshold : bool
        是否自动二值化。

    Returns
    -------
    pandas.DataFrame
        以行列索引组织的 OCR 结果表格。

    Raises
    ------
    ImportError
        如果 pandas 未安装。
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("需要安装 pandas: pip install pandas") from None

    results = ocr_cells(cells, lang=lang, config=config, threshold=threshold)

    if not results:
        return pd.DataFrame()

    max_row = max(r for r, _ in results.keys()) + 1
    max_col = max(c for _, c in results.keys()) + 1

    data: list[list[str]] = [[""] * max_col for _ in range(max_row)]
    for (row, col), text in results.items():
        data[row][col] = text

    return pd.DataFrame(data)
