# QOL for Algo Dev

All in one of QOL for Algo Developer

## Quick Start
### Agent skill installation(Recommend)
Install and use with LLM CLI(it will install the package automatically)

- Gemini Skill Installation
```bash
gemini skills install https://github.com/mightycatty/algo_qol.git --path agent/skills/algo-qol
```
- Other CLI Installation

`TODO`

### PIP Installation
```bash
pip install -r requirement.txt
pip install -e .
```
### Usage Example
**Gemini CLI**
```bash
gemini -p "Use algo-qol to reformat all images in the ./images folder to JPG format."
gemini -p "Use algo-qol to split the YOLO dataset at ./yolo_dataset.txt into training and validation sets with a 9:1 ratio."
```

**CMD CLI**
```bash
> algo-qol --help

# Usage: algo-qol [OPTIONS] COMMAND [ARGS]...
#
#╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
#│ --install-completion          Install completion for the current shell.                                                                                   │
#│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                            │
#│ --help                        Show this message and exit.                                                                                                 │
#╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
#╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
#│ coco-analyse            Analyzes a COCO detection format dataset and returns a dictionary containing various statistics.                                  │
#│ coco-split                                                                                                                                                │
#│ convert-image-format    convert all images in a folder to dst format, support sub-folders                                                                 │
#│ download                download images or videos from give txt or csv                                                                                    │
#│ merge-files             merge all files in a folder(include sub-folders) to a new folder                                                                  │
#│ remove-yolo-none-pair                                                                                                                                     │
#│ rename-files            rename all images in a folder to uuid new name                                                                                    │
#│ rm-corrupted-image      remove corrupted images                                                                                                           │
#│ rm-duplicated-image                                                                                                                                       │
#│ yolo2coco                                                                                                                                                 │
#│ yolov8-to-coco          parse ultrality yolov8 data.yaml to prepare a coco dataset                                                                        │
#╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
**Import as package**
```python
# e.g.
from algo_qol.algo.det.yolo.yolo_utils import yolo_analyse
```

## Reference
[CLI API](./docs/cli.md)