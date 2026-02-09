"""
命令行接口

提供 ``ocr-toolkit`` 命令行工具入口。
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

from ocr_toolkit.pipeline import OCRPipeline, PipelineConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocr-toolkit",
        description="基于 OpenCV + Tesseract 的表格图像 OCR 工具",
    )

    parser.add_argument(
        "input",
        nargs="+",
        help="输入图像路径或通配符模式（如 images/*.jpg）",
    )
    parser.add_argument(
        "-o", "--output",
        default="output",
        help="输出目录路径（默认: output）",
    )
    parser.add_argument(
        "-l", "--lang",
        default="chi_sim+eng",
        help="Tesseract 语言包（默认: chi_sim+eng）",
    )
    parser.add_argument(
        "--format",
        choices=["text", "csv", "excel", "json"],
        default="text",
        help="输出格式（默认: text）",
    )
    parser.add_argument(
        "--no-red-removal",
        action="store_true",
        help="禁用红色去除",
    )
    parser.add_argument(
        "--no-sharpen",
        action="store_true",
        help="禁用锐化",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="禁用降噪",
    )
    parser.add_argument(
        "--no-skew-correct",
        action="store_true",
        help="禁用倾斜校正",
    )
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="启用直方图均衡化增强",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="保存中间处理结果",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract PSM 模式（默认: 6）",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细日志",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口。"""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 展开通配符
    image_paths: list[str] = []
    for pattern in args.input:
        expanded = glob.glob(pattern)
        if expanded:
            image_paths.extend(expanded)
        elif Path(pattern).exists():
            image_paths.append(pattern)
        else:
            logging.warning("未找到匹配: %s", pattern)

    if not image_paths:
        logging.error("没有找到任何输入图像")
        return 1

    # 构建配置
    config = PipelineConfig(
        remove_red=not args.no_red_removal,
        sharpen=not args.no_sharpen,
        denoise=not args.no_denoise,
        correct_skew=not args.no_skew_correct,
        enhance=args.enhance,
        lang=args.lang,
        ocr_config=f"--psm {args.psm}",
        save_intermediates=args.save_intermediates,
        output_dir=args.output,
    )

    pipeline = OCRPipeline(config)

    # 执行处理
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        logging.info("=" * 60)
        logging.info("处理: %s", image_path)
        logging.info("=" * 60)

        try:
            result = pipeline.run(image_path)
        except Exception as e:
            logging.error("处理失败: %s", e)
            continue

        stem = Path(image_path).stem

        if args.format == "text":
            print(f"\n{'=' * 40}")
            print(f"文件: {image_path}")
            print(f"{'=' * 40}")
            print(result.full_text)
            # 同时保存到文件
            text_path = output_dir / f"{stem}.txt"
            text_path.write_text(result.full_text, encoding="utf-8")
            logging.info("文本已保存: %s", text_path)

        elif args.format == "csv":
            csv_path = output_dir / f"{stem}.csv"
            result.to_csv(str(csv_path))
            logging.info("CSV 已保存: %s", csv_path)

        elif args.format == "excel":
            excel_path = output_dir / f"{stem}.xlsx"
            result.to_excel(str(excel_path))
            logging.info("Excel 已保存: %s", excel_path)

        elif args.format == "json":
            import json

            json_data = {
                "file": image_path,
                "has_table": result.has_table,
                "cells": {
                    f"{r},{c}": text for (r, c), text in result.ocr_results.items()
                },
                "full_text": result.full_text,
            }
            json_path = output_dir / f"{stem}.json"
            json_path.write_text(
                json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logging.info("JSON 已保存: %s", json_path)

    logging.info("全部处理完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
