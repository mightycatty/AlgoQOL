# QOL for Algo Dev

All in one of QOL for Algo Developer
## Installation
```bash
pip install -r requirement.txt
pip install -e .
```
## usage
Use as CLI
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
Import as package
```python
# example, when to use yolo utilities
from algo_qol.data.detection.yolo_utils import *
```

## TODO
- [ ] improve CLI respond speed