# QOL for Algo Dev

All in one of QOL for Algo Developer
## Installation
```bash
pip install -r requirement.txt
pip install -e .
```
## usage
Us as CLI
```bash
> algo_qol --help # list all available command group
#
# Usage: algo-qol [OPTIONS] COMMAND [ARGS]...
#
#╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
#│ --install-completion          Install completion for the current shell.                                                                                                                                                          │
#│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                                                                                                   │
#│ --help                        Show this message and exit.                                                                                                                                                                        │
#╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
#╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
#│ cls     CLI toolkit for classification                                                                                                                                                                                           │
#│ det     CLI toolkit for detection                                                                                                                                                                                                │
#│ seg     CLI toolkit for segmentation                                                                                                                                                                                             │
#│ utils   CLI toolkit for general algo dev utilities                                                                                                                                                                               │
#╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
> algo_qol utils --help # list all available commands under a group
# Usage: algo-qol utils [OPTIONS] COMMAND [ARGS]...
#
# CLI toolkit for general algo dev utilities
#
#╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
#│ --help          Show this message and exit.                                                                                                                                                                                      │
#╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
#╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
#│ convert-image-format   convert all images in a folder to dst format, support sub-folders                                                                                                                                         │
#│ merge-files            merge all files in a folder(include sub-folders) to a new folder                                                                                                                                          │
#│ rename-files           rename all images in a folder to uuid new name                                                                                                                                                            │
#│ rm-corrupted-image     remove corrupted images Args:     src_f: src folder， support sub folders     threads_num:     remove: whether to remove or move to a corrupted folder                                                    │
#│ rm-duplicated-image                                                                                                                                                                                                              │
#╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
Import as package
```python
# example, when to use yolo utilities
from algo_qol.data.detection.yolo_utils import *
```
