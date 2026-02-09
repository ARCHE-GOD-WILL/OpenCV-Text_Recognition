"""
ocr_toolkit - 基于 OpenCV + Tesseract 的表格图像 OCR 工具包

功能:
    - 图像预处理（去色、锐化、降噪、直方图均衡化）
    - 倾斜校正（自动检测并纠正文档倾斜）
    - 表格线条检测（水平线 + 垂直线）
    - 单元格切片（根据检测到的线条将表格切割为单元格）
    - OCR 文字识别（基于 Tesseract）
    - 完整流水线（从原始图像到结构化文本输出）
"""

__version__ = "1.0.0"
__author__ = "OCR Toolkit Contributors"

from ocr_toolkit.preprocessing import (
    remove_red_color,
    sharpen_image,
    denoise_image,
    enhance_image,
    preprocess_image,
)
from ocr_toolkit.skew import correct_skew
from ocr_toolkit.line_detect import detect_horizontal_lines, detect_vertical_lines, detect_all_lines
from ocr_toolkit.slicer import slice_by_lines, slice_grid
from ocr_toolkit.ocr import ocr_image, ocr_image_to_dataframe
from ocr_toolkit.pipeline import OCRPipeline

__all__ = [
    "remove_red_color",
    "sharpen_image",
    "denoise_image",
    "enhance_image",
    "preprocess_image",
    "correct_skew",
    "detect_horizontal_lines",
    "detect_vertical_lines",
    "detect_all_lines",
    "slice_by_lines",
    "slice_grid",
    "ocr_image",
    "ocr_image_to_dataframe",
    "OCRPipeline",
]
