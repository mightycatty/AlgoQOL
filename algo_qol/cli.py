import typer
from algo_qol.data.utils.convert_img_format import convert_image_format
from algo_qol.data.utils.remove_corrupted_images import rm_corrupted_image
from algo_qol.data.utils.remove_duplicated_imgs import rm_duplicated_image
from algo_qol.data.utils.rename_img import rename_files
from algo_qol.data.utils.merge_files import merge_files
from algo_qol.data.detection.yolo_utils import remove_yolo_none_pair

app = typer.Typer()

utils_typer = typer.Typer()

app.add_typer(utils_typer, name='utils', help='CLI toolkit for general algo dev utilities')
# commands group
utils_typer.command()(convert_image_format)
utils_typer.command()(rm_corrupted_image)
utils_typer.command()(rm_duplicated_image)
utils_typer.command()(rename_files)
utils_typer.command()(merge_files)

typer_group = typer.Typer()
app.add_typer(typer_group, name='det', help='CLI toolkit for detection')
typer_group.command()(remove_yolo_none_pair)

typer_group = typer.Typer()
app.add_typer(typer_group, name='seg', help='CLI toolkit for segmentation')
typer_group.command()(remove_yolo_none_pair)

typer_group = typer.Typer()
app.add_typer(typer_group, name='cls', help='CLI toolkit for classification')
typer_group.command()(remove_yolo_none_pair)


def main():
    app()


if __name__ == '__main__':
    main()
