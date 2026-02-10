# OCR Toolkit - 表格图像文字识别工具包

基于 **OpenCV** + **Tesseract OCR** 的纯 Python 表格图像文字识别工具包。

能够自动完成图像预处理、倾斜校正、表格线条检测、单元格切割和 OCR 文字识别的完整流水线，将表格图像转换为结构化文本数据。

## 功能特性

| 功能 | 说明 |
|------|------|
| 图像预处理 | 红色印章/线条去除、锐化、降噪、直方图均衡化 |
| 倾斜校正 | 自动检测文档倾斜角度并矫正 |
| 表格检测 | 基于 Hough 变换的水平线和垂直线检测 |
| 单元格切割 | 根据检测到的线条将表格图像切割为独立单元格 |
| OCR 识别 | 基于 Tesseract 引擎，支持中文（简/繁）和英文 |
| 多格式输出 | 纯文本、CSV、Excel、JSON |
| 批量处理 | 支持通配符批量处理多张图像 |

## 处理流水线

```
原始图像 → 去除红色 → 锐化 → 降噪 → 倾斜校正 → 线条检测 → 单元格切割 → OCR识别 → 结构化输出
```

## 安装

### 前置依赖

需要先安装 Tesseract OCR 引擎：

**macOS:**

```bash
brew install tesseract tesseract-lang
```

**Ubuntu / Debian:**

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra
```

**Windows:**

从 [Tesseract 官方](https://github.com/UB-Mannheim/tesseract/wiki) 下载安装包，安装时勾选中文语言包。

### 安装工具包

```bash
# 基础安装
pip install -e .

# 包含 pandas 和 Excel 支持
pip install -e ".[all]"

# 或直接使用 requirements.txt
pip install -r requirements.txt
```

## 快速开始

### 命令行使用

```bash
# 基本用法 - 识别单张图像
ocr-toolkit image.jpg

# 指定中文识别，输出为 CSV
ocr-toolkit image.jpg -l chi_sim+eng --format csv

# 批量处理，保存中间结果
ocr-toolkit images/*.jpg -o results --save-intermediates

# 输出为 Excel
ocr-toolkit table.png --format excel -o output

# 查看帮助
ocr-toolkit --help
```

### Python API 使用

#### 完整流水线

```python
from ocr_toolkit import OCRPipeline
from ocr_toolkit.pipeline import PipelineConfig

# 使用默认配置
pipeline = OCRPipeline()
result = pipeline.run("table_image.jpg")

# 输出识别文本
print(result.full_text)

# 导出为 CSV
result.to_csv("output.csv")

# 导出为 Excel
result.to_excel("output.xlsx")

# 转为 pandas DataFrame
df = result.to_dataframe()
print(df)
```

#### 自定义配置

```python
from ocr_toolkit.pipeline import OCRPipeline, PipelineConfig

config = PipelineConfig(
    remove_red=True,          # 去除红色印章
    sharpen=True,             # 锐化
    denoise=True,             # 降噪
    correct_skew=True,        # 倾斜校正
    lang="chi_sim+eng",       # 中英文混合识别
    save_intermediates=True,  # 保存中间结果
    output_dir="my_output",   # 输出目录
)

pipeline = OCRPipeline(config)
result = pipeline.run("document.jpg")
```

#### 单独使用各模块

```python
import cv2
from ocr_toolkit import (
    remove_red_color,
    sharpen_image,
    denoise_image,
    correct_skew,
    detect_all_lines,
    slice_grid,
    ocr_image,
)

# 读取图像
image = cv2.imread("table.jpg")

# 1. 预处理
clean = remove_red_color(image)
sharp = sharpen_image(clean)
denoised = denoise_image(sharp)

# 2. 倾斜校正
corrected = correct_skew(denoised)

# 3. 检测表格线
lines = detect_all_lines(corrected)
print(f"水平线: {lines.horizontal}")
print(f"垂直线: {lines.vertical}")

# 4. 切割单元格
cells = slice_grid(corrected, detected_lines=lines)

# 5. OCR 识别
for cell in cells:
    text = ocr_image(cell.image, lang="chi_sim+eng")
    print(f"单元格 [{cell.row}, {cell.col}]: {text}")
```

## 项目结构

```
ocr_toolkit/
├── __init__.py         # 包入口，导出公开 API
├── preprocessing.py    # 图像预处理（去色、锐化、降噪、增强）
├── skew.py             # 倾斜检测与校正
├── line_detect.py      # 水平线/垂直线检测
├── slicer.py           # 图像切割（行条带、网格单元格）
├── ocr.py              # Tesseract OCR 文字识别
├── pipeline.py         # 完整流水线与结果导出
└── cli.py              # 命令行接口
```

## CLI 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入图像路径（支持通配符） | 必填 |
| `-o, --output` | 输出目录 | `output` |
| `-l, --lang` | Tesseract 语言包 | `chi_sim+eng` |
| `--format` | 输出格式: text/csv/excel/json | `text` |
| `--no-red-removal` | 禁用红色去除 | - |
| `--no-sharpen` | 禁用锐化 | - |
| `--no-denoise` | 禁用降噪 | - |
| `--no-skew-correct` | 禁用倾斜校正 | - |
| `--enhance` | 启用直方图均衡化 | - |
| `--save-intermediates` | 保存中间处理结果 | - |
| `--psm` | Tesseract PSM 模式 | `6` |
| `-v, --verbose` | 显示详细日志 | - |

## Tesseract PSM 模式参考

| PSM | 说明 | 适用场景 |
|-----|------|----------|
| 3 | 全自动页面分割 | 通用文档 |
| 6 | 统一文本块 | 表格单元格（默认） |
| 7 | 单行文本 | 单行数据 |
| 8 | 单词 | 短文本/数字 |
| 13 | 原始行 | 不规则文本 |

## 技术栈

- **Python** 3.12+
- **OpenCV** - 图像处理核心
- **Tesseract OCR** (pytesseract) - 文字识别引擎
- **NumPy** - 数值计算
- **pandas** (可选) - 表格数据处理
- **openpyxl** (可选) - Excel 导出

## 许可证

MIT License

## Web UI (Streamlit)

Provide a visual interface for adjusting parameters and verifying results.

### Launch

```bash
streamlit run app.py
```

### Features

- **Visual Debugging**: See detected lines and intermediate processing steps.
- **Parameter Tuning**: Adjust Canny/Hough thresholds in real-time.
- **Data Export**: Download OCR results as CSV.
- **Structure Visualization**: Verify if table rows/columns are correctly identified.
