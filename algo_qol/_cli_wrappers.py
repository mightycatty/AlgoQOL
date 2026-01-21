'''Lightweight Typer command wrappers that defer heavy imports until execution.'''
from __future__ import annotations

from typing import Optional

import typer

app = typer.Typer(help='Utility shortcuts for algorithm QOL workflows.')


@app.command()
def image_reformat(
        src_f: str = typer.Argument(..., help='Root directory containing images (recursively scanned).'),
        dst_format: str = typer.Option(
            'jpg',
            '--dst-format',
            '-f',
            help='Destination image format (e.g. jpg, png, webp).',
        ),
        threads_num: int = typer.Option(
            16,
            '--workers',
            '-w',
            min=1,
            help='Number of parallel workers used for conversion.',
        ),
):
    '''Convert every image under *src_f* into a new format.'''
    from algo_qol.data_utils.file_utils import convert_image_format as _impl
    return _impl(src_f=src_f, dst_format=dst_format, threads_num=threads_num)


@app.command()
def image_cleanup(
        src_f: str = typer.Argument(..., help='Directory that will be scanned for unreadable images.'),
        threads_num: int = typer.Option(128, '--workers', '-w', min=1, help='Thread pool size.'),
        move_out: bool = typer.Option(
            False,
            '--move-out/--delete',
            help="Move corrupted files to a 'corrupted' folder instead of deleting them.",
        ),
):
    '''Remove or relocate corrupted images under *src_f*.'''
    from algo_qol.data_utils.file_utils import rm_corrupted_image as _impl

    return _impl(src_f=src_f, threads_num=threads_num, move_out=move_out)


@app.command()
def rename_files(
        src_f: str = typer.Argument(..., help='Directory whose images will be renamed with UUIDs.'),
):
    '''Rename every image inside *src_f* with a freshly generated UUID.'''
    from algo_qol.data_utils.file_utils import rename_files as _impl
    return _impl(src_f)


@app.command('download')
def download_assets(
        source: str = typer.Argument(
            ...,
            help='Path to a txt/csv file containing URLs (one per line or column).',
        ),
        save_f: str = typer.Argument(..., help='Directory where downloaded assets are saved.'),
        thread_num: int = typer.Option(16, '--workers', '-w', min=1, help='Concurrent download threads.'),
        col_index_or_name: Optional[str] = typer.Option(
            None,
            '--column',
            '-c',
            help='Column index or header name for URLs when reading a CSV file.',
        ),
        remove_http_head: bool = typer.Option(
            True,
            '--strip-host/--keep-host',
            help='Strip the protocol/host portion when generating filenames.',
        ),
):
    '''Multi-threaded downloader for image or video URLs.'''
    from algo_qol.data_utils.video_img_downloader import download as _impl

    index = col_index_or_name
    if index is None:
        index = 0
    else:
        try:
            index = int(index)
        except ValueError:
            pass
    return _impl(
        txt_or_csv_or_list=source,
        save_f=save_f,
        thread_num=thread_num,
        col_index_or_name=index,
        remove_http_head=remove_http_head,
    )


@app.command()
def image_dedup(
        root: str = typer.Argument(..., help='Directory containing images to process.'),
        method: str = typer.Option('clip', help="Deduplication backend: 'clip' or 'phash'."),
        similarity: Optional[float] = typer.Option(
            0.98,
            help='Similarity threshold (cosine for CLIP, mapped to distance for pHash).',
        ),
        distance: Optional[int] = typer.Option(10, help='pHash Hamming distance threshold.'),
        model: str = typer.Option('google/siglip-base-patch16-224', help='CLIP/SigLIP model identifier.'),
        batch_size: int = typer.Option(256, help='Per-device batch size for feature extraction.'),
        cpu_index: bool = typer.Option(False, help='Force FAISS to run on CPU.'),
        skip_gallery_vis: bool = typer.Option(False, help='Skip duplicate gallery visualisation (CLIP only).'),
        move_out: Optional[str] = typer.Option(
            None,
            help='Move duplicates into this folder instead of deleting (implies create).',
        ),
        debug: bool = typer.Option(
            False,
            help='Debug mode: small sample, keep files, dump visual diagnostics.',
        ),
        recursive: bool = typer.Option(True, help='pHash: traverse sub-directories individually.'),
):
    '''Remove duplicate images using CLIP/SigLIP features or perceptual hash.'''
    from algo_qol.data_utils.image_dedup import dedup_images as _impl

    return _impl(
        root=root,
        method=method,
        similarity=similarity,
        distance=distance,
        model=model,
        batch_size=batch_size,
        cpu_index=cpu_index,
        skip_gallery_vis=skip_gallery_vis,
        move_out=move_out,
        debug=debug,
        recursive=recursive,
    )


@app.command()
def image_resize(
        img_src: str = typer.Argument(..., help='Directory containing images to resize.'),
        img_dst: Optional[str] = typer.Option(
            None,
            '--dst',
            help='Optional output directory (defaults to in-place modification).',
        ),
        max_size: int = typer.Option(720, help='Target maximum dimension for each image.'),
        max_workers: int = typer.Option(16, '--workers', '-w', min=1, help='Parallel workers.'),
):
    '''Resize images so that the longest side equals *max_size* pixels.'''
    from algo_qol.data_utils.file_utils import resize_image as _impl

    return _impl(img_src=img_src, img_dst=img_dst, max_size=max_size, max_workers=max_workers)


@app.command()
def sample_files(
        src_f: str = typer.Argument(..., help='Source directory to sample files from (recursively).'),
        dst_f: str = typer.Argument(..., help='Destination directory to save sampled files.'),
        sample_num: int = typer.Option(1000, help='Number of files to sample.'),
        threads_num: int = typer.Option(16, '--workers', '-w', min=1, help='Number of parallel workers.'),
):
    '''Recursively sample files from *src_f* and copy them to *dst_f*.'''
    from algo_qol.data_utils.file_utils import sample_files as _impl
    return _impl(src_f=src_f, dst_f=dst_f, sample_num=sample_num, threads_num=threads_num)


@app.command()
def remove_yolo_none_pair(
        img_f: str = typer.Argument(..., help='Directory containing YOLO-format images.'),
        label_f: Optional[str] = typer.Option(
            None,
            '--label-dir',
            help="Label directory (defaults to replacing 'images' with 'labels').",
        ),
):
    '''Move negative samples (empty labels) to a separate folder.'''
    from algo_qol.algo.det.yolo.yolo_utils import remove_yolo_none_pair as _impl

    return _impl(img_f=img_f, label_f=label_f)


@app.command()
def yolo2coco(
        images_dir: str = typer.Argument(
            ...,
            help='Image directory (or root) to convert. Accepts a single folder path.',
        ),
        labels_dir: str = typer.Argument(..., help='Directory containing YOLO label txt files.'),
        output_json: str = typer.Argument(..., help='Destination COCO annotation json path.'),
        classes_or_txt: str = typer.Argument(
            ...,
            help='Class list (comma separated) or path to a txt file with one class per line.',
        ),
        absolute_path: bool = typer.Option(
            False,
            '--absolute-path/--relative-path',
            help='Store absolute image paths in the resulting COCO json.',
        ),
):
    '''Convert YOLO-format annotations into COCO json.'''
    from algo_qol.algo.det.convertor.yolo2coco import yolo2coco as _impl

    return _impl(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_json=output_json,
        classes_or_txt=classes_or_txt,
        absolute_path=absolute_path,
    )


@app.command()
def yolov8_to_coco(
        yolov8_data_yaml: str = typer.Argument(..., help='Path to ultralytics data.yaml file.'),
        dataset_prefix: str = typer.Argument(..., help='Prefix for generated COCO json files.'),
        absolute_path: bool = typer.Option(
            False,
            '--absolute-path/--relative-path',
            help='Store absolute image paths in COCO outputs.',
        ),
):
    '''Create COCO annotations from a YOLOv8 dataset configuration.'''
    from algo_qol.algo.det.convertor.yolo2coco import yolov8_to_coco as _impl

    return _impl(
        yolov8_data_yaml=yolov8_data_yaml,
        dataset_prefix=dataset_prefix,
        absolute_path=absolute_path,
    )


@app.command()
def coco_split(
        coco_file: str = typer.Argument(..., help='Path to the source COCO annotation json.'),
        val_ratio: float = typer.Option(0.1, help='Fraction of images to place in the validation split.'),
        seed: int = typer.Option(0, help='Random seed used for shuffling.'),
        print_stats: bool = typer.Option(True, help='Print dataset statistics for each split.'),
):
    '''Split a COCO dataset into train/val json files.'''
    from algo_qol.algo.det.coco.coco_utils import coco_split as _impl

    return _impl(coco_file=coco_file, val_ratio=val_ratio, seed=seed, print_stats=print_stats)


@app.command()
def coco_analyse(
        annotations_file: str = typer.Argument(..., help='Path to a COCO annotation json file.'),
        verbose: bool = typer.Option(True, help='Print the computed statistics.'),
):
    '''Summarise class and bounding-box statistics for a COCO dataset.'''
    from algo_qol.algo.det.coco.coco_utils import coco_analyse as _impl

    return _impl(annotations_file=annotations_file, verbose=verbose)


__all__ = [
    'app',
]