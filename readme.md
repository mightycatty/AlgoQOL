# 🚀 QOL for Algo Dev

All-in-one QOL toolkit for Algorithm Developers 🛠️

## ✨ Features

### 📊 Dataset Management
*   **Analysis & Splitting**: Summarise statistics (`coco-analyse`) and split datasets (`coco-split`).
*   **Format Conversion**: Convert YOLO/YOLOv8 to COCO (`yolo2coco`, `yolov8-to-coco`).
*   **Data Cleaning**: Handle negative samples (`remove-yolo-none-pair`).

### 🖼️ Image Processing
*   **Deduplication**: Remove duplicates using CLIP/SigLIP or pHash (`image-dedup`).
*   **Quality Control**: Identify and remove corrupted images (`image-cleanup`).
*   **Batch Processing**: Resize (`image-resize`) and reformat (`image-reformat`) images efficiently.

### 📂 File Operations
*   **Management**: Multi-threaded downloader (`download`), directory merging (`merge-files`), and UUID renaming (`rename-files`).

### 🤖 Algorithm SDK & Modules

#### 👁️ Computer Vision (CV)
*   **Detection (DET)**:
    *   **Inference Wrappers**: `Yolov8DetTorch` (YOLOv8 inference with custom NMS).
    *   **Format Converters**: COCO <-> YOLO, LabelMe -> YOLO, OCR -> YOLO.
    *   **Utilities**: COCO JSON browsing, YOLO TXT browsing.
*   **OCR**:
    *   **Data Utils**: Train/Val splitter for PaddleOCR datasets.
*   **Classification (CLS)**:
    *   **Metrics**: `cls_report` for calculating Precision, Recall, ROC, and AP.

#### 🧠 Multimodal (LLM/VLM)
*   **VLLM Client**: `VLLMClient` for interacting with OpenAI-compatible APIs (vLLM), supporting Text + Image + Video inputs.
*   **Robust Parsing**:
    *   `parse_llm_json`: Extract and repair JSON from buggy LLM outputs.
    *   `parse_cot_response`: Parse "Chain of Thought" (`<think>...</think>`) responses.
*   **confidence Scoring**:
    *   Calculate confidence scores for Boolean, Integer, or List outputs using Logprobs (`get_bool_value_score`, `get_int_value_score`).

## ⚡ Quick Start

### 🤖 Agent Skill Installation (Recommended)
The agent will automatically discover the correct CLI commands (using `algo-qol --help`), check for installation, and execute the tasks securely.

- **Gemini Skill Installation**
```bash
gemini skills install https://github.com/mightycatty/algo_qol.git --path agent/skills/algo-qol
```
- **Other CLI Installation**

`TODO`

### 📦 PIP Installation
```bash
pip install -r requirement.txt
pip install -e .
```

### 💡 Usage Example

**🧠 Gemini CLI**
```bash
gemini -p "Use algo-qol to reformat all images in the ./images folder to JPG format."
gemini -p "Use algo-qol to split the YOLO dataset at ./yolo_dataset.txt into training and validation sets with a 9:1 ratio."
```

**💻 CMD CLI**
```bash
> algo-qol --help

# Usage: algo-qol [OPTIONS] COMMAND [ARGS]...
# ...
```

**🐍 Import as package**
```python
# e.g.
from algo_qol.algo.det.yolo.yolo_utils import yolo_analyse
```

## 📚 Reference
[CLI API](./docs/cli.md)