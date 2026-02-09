"""
完整 OCR 流水线模块

将预处理、倾斜校正、线条检测、切片、OCR 识别整合为一条流水线。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from ocr_toolkit.line_detect import DetectedLines, detect_all_lines
from ocr_toolkit.ocr import ocr_cells, ocr_image
from ocr_toolkit.preprocessing import preprocess_image
from ocr_toolkit.skew import correct_skew
from ocr_toolkit.slicer import CellImage, save_cells, slice_grid

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """流水线配置参数。"""

    # 预处理
    remove_red: bool = True
    sharpen: bool = True
    denoise: bool = True
    enhance: bool = False
    sharpen_kernel_size: int = 40
    denoise_strength: int = 10

    # 倾斜校正
    correct_skew: bool = True

    # 线条检测
    h_canny_low: int = 50
    h_canny_high: int = 150
    h_hough_threshold: int = 150
    h_min_line_length: int = 120
    v_canny_low: int = 20
    v_canny_high: int = 40
    v_hough_threshold: int = 10
    line_min_gap: int = 40

    # OCR
    lang: str = "chi_sim+eng"
    ocr_config: str = "--psm 6"
    ocr_threshold: bool = True

    # 输出
    save_intermediates: bool = False
    output_dir: str = "output"


@dataclass
class PipelineResult:
    """流水线执行结果。"""

    original_image: NDArray[np.uint8]
    preprocessed_image: NDArray[np.uint8]
    corrected_image: NDArray[np.uint8] | None = None
    detected_lines: DetectedLines | None = None
    cells: list[CellImage] = field(default_factory=list)
    ocr_results: dict[tuple[int, int], str] = field(default_factory=dict)
    full_text: str = ""

    @property
    def has_table(self) -> bool:
        """是否检测到表格结构。"""
        if self.detected_lines is None:
            return False
        return len(self.detected_lines.horizontal) > 0 or len(self.detected_lines.vertical) > 0

    def to_dataframe(self):
        """将 OCR 结果转为 pandas DataFrame。"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("需要安装 pandas: pip install pandas") from None

        if not self.ocr_results:
            return pd.DataFrame()

        max_row = max(r for r, _ in self.ocr_results.keys()) + 1
        max_col = max(c for _, c in self.ocr_results.keys()) + 1

        data: list[list[str]] = [[""] * max_col for _ in range(max_row)]
        for (row, col), text in self.ocr_results.items():
            data[row][col] = text

        return pd.DataFrame(data)

    def to_csv(self, path: str, encoding: str = "utf-8-sig") -> None:
        """将 OCR 结果保存为 CSV 文件。"""
        df = self.to_dataframe()
        df.to_csv(path, index=False, header=False, encoding=encoding)

    def to_excel(self, path: str) -> None:
        """将 OCR 结果保存为 Excel 文件。"""
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            raise ImportError("需要安装 openpyxl: pip install openpyxl") from None

        df = self.to_dataframe()
        df.to_excel(path, index=False, header=False)


class OCRPipeline:
    """完整的 OCR 处理流水线。

    Examples
    --------
    >>> from ocr_toolkit import OCRPipeline
    >>> pipeline = OCRPipeline()
    >>> result = pipeline.run("table_image.jpg")
    >>> print(result.full_text)
    >>> result.to_csv("output.csv")
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def run(self, image_path: str) -> PipelineResult:
        """执行完整的 OCR 流水线。

        Parameters
        ----------
        image_path : str
            输入图像路径。

        Returns
        -------
        PipelineResult
            包含所有中间结果和最终 OCR 文本的结果对象。

        Raises
        ------
        FileNotFoundError
            图像文件不存在。
        ValueError
            无法读取图像。
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        original = cv2.imread(str(path))
        if original is None:
            raise ValueError(f"无法读取图像: {image_path}")

        return self.run_on_image(original, source_name=path.stem)

    def run_on_image(
        self, image: NDArray[np.uint8], source_name: str = "image"
    ) -> PipelineResult:
        """对内存中的图像执行完整流水线。

        Parameters
        ----------
        image : ndarray
            BGR 格式输入图像。
        source_name : str
            图像来源名称（用于保存中间结果）。

        Returns
        -------
        PipelineResult
            处理结果。
        """
        cfg = self.config
        output_dir = os.path.join(cfg.output_dir, source_name)

        logger.info("开始处理图像: %s", source_name)

        # 1. 预处理
        logger.info("步骤 1/5: 图像预处理")
        preprocessed = preprocess_image(
            image,
            remove_red=cfg.remove_red,
            sharpen=cfg.sharpen,
            denoise=cfg.denoise,
            enhance=cfg.enhance,
            sharpen_kernel_size=cfg.sharpen_kernel_size,
            denoise_strength=cfg.denoise_strength,
        )

        if cfg.save_intermediates:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, "01_preprocessed.jpg"), preprocessed)

        # 2. 倾斜校正
        corrected = None
        current = preprocessed
        if cfg.correct_skew:
            logger.info("步骤 2/5: 倾斜校正")
            corrected = correct_skew(preprocessed)
            current = corrected
            if cfg.save_intermediates:
                cv2.imwrite(os.path.join(output_dir, "02_corrected.jpg"), corrected)

        # 3. 线条检测
        logger.info("步骤 3/5: 线条检测")
        detected = detect_all_lines(
            current,
            h_canny_low=cfg.h_canny_low,
            h_canny_high=cfg.h_canny_high,
            h_hough_threshold=cfg.h_hough_threshold,
            h_min_line_length=cfg.h_min_line_length,
            v_canny_low=cfg.v_canny_low,
            v_canny_high=cfg.v_canny_high,
            v_hough_threshold=cfg.v_hough_threshold,
            min_gap=cfg.line_min_gap,
        )
        logger.info(
            "检测到 %d 条水平线, %d 条垂直线",
            len(detected.horizontal),
            len(detected.vertical),
        )

        if cfg.save_intermediates:
            debug_img = current.copy()
            for y in detected.horizontal:
                cv2.line(debug_img, (0, y), (detected.image_width, y), (0, 0, 255), 2)
            for x in detected.vertical:
                cv2.line(debug_img, (x, 0), (x, detected.image_height), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_dir, "03_lines.jpg"), debug_img)

        # 4. 切片
        logger.info("步骤 4/5: 图像切片")
        cells = slice_grid(current, detected_lines=detected)
        logger.info("切割出 %d 个单元格", len(cells))

        if cfg.save_intermediates:
            cells_dir = os.path.join(output_dir, "cells")
            save_cells(cells, cells_dir)

        # 5. OCR 识别
        logger.info("步骤 5/5: OCR 文字识别")
        if cells:
            ocr_results = ocr_cells(
                cells, lang=cfg.lang, config=cfg.ocr_config, threshold=cfg.ocr_threshold
            )
        else:
            # 没有检测到表格，对整张图片做 OCR
            full_text = ocr_image(
                current, lang=cfg.lang, config=cfg.ocr_config, threshold=cfg.ocr_threshold
            )
            ocr_results = {(0, 0): full_text}

        # 组合全文
        full_text_parts: list[str] = []
        sorted_keys = sorted(ocr_results.keys())
        current_row = -1
        for row, col in sorted_keys:
            if row != current_row:
                if current_row >= 0:
                    full_text_parts.append("\n")
                current_row = row
            text = ocr_results[(row, col)]
            if text:
                full_text_parts.append(text)
                full_text_parts.append("\t")

        full_text = "".join(full_text_parts).strip()
        logger.info("OCR 完成，识别出 %d 个字符", len(full_text))

        return PipelineResult(
            original_image=image,
            preprocessed_image=preprocessed,
            corrected_image=corrected,
            detected_lines=detected,
            cells=cells,
            ocr_results=ocr_results,
            full_text=full_text,
        )

    def run_batch(self, image_paths: list[str]) -> list[PipelineResult]:
        """批量处理多张图像。

        Parameters
        ----------
        image_paths : list[str]
            图像路径列表。

        Returns
        -------
        list[PipelineResult]
            结果列表。
        """
        results: list[PipelineResult] = []
        for i, path in enumerate(image_paths, 1):
            logger.info("处理进度: %d/%d - %s", i, len(image_paths), path)
            try:
                result = self.run(path)
                results.append(result)
            except Exception as e:
                logger.error("处理 %s 失败: %s", path, e)
        return results
