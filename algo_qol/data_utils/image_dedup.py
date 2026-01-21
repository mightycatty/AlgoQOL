#!/usr/bin/env python3
"""
Image Deduplication (Unified): CLIP/SigLIP or pHash via one interface.

Examples:
  # CLIP with default similarity=0.95
  python image_dedup.py dedup_images --root /path/to/images --method clip

  # CLIP with similarity threshold
  python image_dedup.py dedup_images --root /path/to/images --method clip --similarity 0.93

  # pHash with integer Hamming distance
  python image_dedup.py dedup_images --root /path/to/images --method phash --distance 6

  # pHash but user thinks in similarity: map to distance internally
  python image_dedup.py dedup_images --root /path/to/images --method phash --similarity 0.9

Notes on parameter compatibility:
- CLIP uses cosine similarity threshold in [0,1].
- pHash uses Hamming distance threshold on a 64-bit hash.
  If only `similarity` is provided, we convert: distance ≈ round((1 - similarity) * 64).
"""
import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

import contextlib
import os
import random
import shutil
import uuid
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional

import PIL
import cv2
import faiss
import fire
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from imagededup.methods import PHash
from loguru import logger
from skimage import io
from tqdm import tqdm

# ---------------------------------------
# Globals
# ---------------------------------------
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif", 'jfif'}

_OPENAI_CLIP = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-large-patch14-336",
]

_SIGLIP_MODEL = [
    "google/siglip-base-patch16-224",
    "google/siglip-so400m-patch14-384",
]

_PHASH_BITS = 64  # imagededup PHash length


# ---------------------------------------
# CLIP backbone (unchanged core)
# ---------------------------------------
class CLIP(torch.nn.Module):
    def __init__(self, model_name='openai/clip-vit-large-patch14-336'):
        super().__init__()
        from transformers import CLIPProcessor, CLIPVisionModel, SiglipVisionModel, SiglipProcessor
        if model_name in _OPENAI_CLIP:
            self.model = CLIPVisionModel.from_pretrained(model_name)
            self._processor = CLIPProcessor.from_pretrained(model_name)
        elif model_name in _SIGLIP_MODEL:
            self.model = SiglipVisionModel.from_pretrained(model_name)
            self._processor = SiglipProcessor.from_pretrained(model_name)
        else:
            raise NotImplementedError(f'unsupported model: {model_name}')

    def processor(self, images: PIL.Image, **kwargs):
        data = self._processor(images=images, return_tensors="pt")
        data['pixel_values'] = torch.squeeze(data['pixel_values'])
        return data

    def forward(self, **kwargs):
        x = self.model(output_hidden_states=True, **kwargs)
        pooler = x.pooler_output
        patch_embeddings = x.hidden_states[4][:, 1:, :]
        mean_0 = patch_embeddings.mean(dim=-1)
        patch_embeddings = x.hidden_states[-4][:, 1:, :]
        mean_1 = patch_embeddings.mean(dim=-1)
        z = torch.cat([pooler, mean_0, mean_1], dim=-1)
        return z


def init_model_preprocessor(model_name: str, device: str) -> Tuple[torch.nn.Module, Any]:
    if model_name in _OPENAI_CLIP or model_name in _SIGLIP_MODEL:
        model = CLIP(model_name)
        processor = model.processor
    else:
        raise NotImplementedError(f'unsupported model: {model_name}')
    model = model.eval()
    # if device == "cuda" and torch.cuda.device_count() > 1:
    #     logger.info(f"Using {torch.cuda.device_count()} GPUs for inference")
    #     model = torch.nn.DataParallel(model)
    model = model.to(device)
    return model, processor


def extract_features(data, model, device: str, norm: bool = True) -> np.ndarray:
    amp_ctx = torch.cuda.amp.autocast() if device == "cuda" else contextlib.nullcontext()
    with torch.no_grad(), amp_ctx:
        data = {k: v if not isinstance(v, torch.Tensor) else v.to(device, non_blocking=True) for k, v in data.items()}
        z = model(**data)
        if norm:
            z = z / z.norm(dim=-1, keepdim=True)
    return z.cpu().numpy().astype(np.float32)


def list_images(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    if extensions is None:
        extensions = list(_IMAGE_EXTS)
    image_files = []
    root_dir = Path(directory)
    for path in root_dir.rglob('*'):
        if path.is_file() and path.suffix.lower() in extensions:
            image_files.append(str(path))
    return image_files


class ImgDS(torch.utils.data.Dataset):
    def __init__(self, paths: List[Path], tfm):
        self.paths = paths
        self.tfm = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except OSError:
            logger.error(f'error when reading {str(self.paths[idx])}')
            return None, str(self.paths[idx])
        data = self.tfm(images=img, return_tensors="pt")
        return data, str(self.paths[idx])


def custom_collate_fn(batch):
    filtered = [(d, p) for d, p in batch if d is not None]
    if not filtered:
        return None
    return torch.utils.data.default_collate(filtered)


def build_faiss_index(mat: np.ndarray, gpu: bool = True) -> faiss.Index:
    d = mat.shape[1]
    index = faiss.IndexFlatIP(d)
    if gpu and faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(mat)
    return index


def group_duplicates_cosine(index, feats: np.ndarray, files: List[str], thr: float) -> List[List[str]]:
    try:
        sims, idxs = index.search(feats, k=min(feats.shape[0], 2048))
    except Exception:
        sims, idxs = index.search(feats, k=min(feats.shape[0], 2048))

    parent = list(range(len(files)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    for i, (nbrs, ss) in enumerate(zip(idxs, sims)):
        for j, s in zip(nbrs[1:], ss[1:]):  # skip self
            if s >= thr:
                union(i, j)

    groups_map: Dict[int, List[str]] = {}
    for i in range(len(files)):
        r = find(i)
        groups_map.setdefault(r, []).append(files[i])
    return list(groups_map.values())


def draw_gallery(index, feats, files: List[str], root: str, num_samples: int = 6, group_size: int = 20):
    random.seed(0)
    if len(feats) == 0:
        return
    sample_indices = random.sample(range(len(feats)), min(num_samples, len(feats)))
    sims, idxs = index.search(feats[sample_indices], k=min(group_size, len(feats)))

    for i, (similarities, indices) in enumerate(zip(sims, idxs)):
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        fig.suptitle(f"Sample {i + 1}: Similar Images", fontsize=20, fontweight='bold')

        for j, (sim, idx) in enumerate(zip(similarities, indices)):
            img_path = files[idx]
            img = Image.open(img_path).convert("RGB")

            # resize to max 512
            max_size = 512
            w, h = img.size
            if max(w, h) > max_size:
                r = max_size / max(w, h)
                img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)

            draw = ImageDraw.Draw(img)
            iw, ih = img.size
            font_size = max(24, int(min(iw, ih) / 5))
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

            text = f"Sim: {sim:.2f}"
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            tw, th = right - left, bottom - top
            x = iw - tw - 20
            y = ih - th - 10
            pad = 10
            draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=(0, 0, 0, 192))
            draw.text((x, y), text, fill="white", font=font)

            ax = axes[j // 5, j % 5]
            ax.imshow(np.array(img))
            ax.axis('off')
            if j == 0:
                ax.set_title("Reference", color='red', fontsize=14, fontweight='bold')

        # remove empty subplots
        for j in range(group_size, 20):
            try:
                fig.delaxes(axes[j // 5, j % 5])
            except Exception:
                break

        plt.tight_layout()
        save_dst = os.path.abspath(os.path.join(root, f"gallery_sample_{i + 1}.png"))
        plt.savefig(save_dst, dpi=300, bbox_inches='tight')
        plt.close()


# ---------------------------------------
# CLIP pipeline
# ---------------------------------------
def _dedup_clip_pipeline(
        root: str,
        similarity: float = 0.95,
        model_name: str = 'google/siglip-base-patch16-224',
        batch_size: int = 256,
        cpu_index: bool = False,
        debug: bool = False,
        move_out: Optional[str] = None,
        skip_gallery_vis: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = list_images(root)
    random.shuffle(paths)
    if debug:
        logger.debug('Debug mode: cap at 10,000 images.')
        paths = paths[:10000]
    logger.info(f'Total images: {len(paths)}')

    feats = np.empty((0, 1), dtype=np.float32)
    files: List[str] = []

    if len(paths) > 0:
        model, processor = init_model_preprocessor(model_name, device)
        effective_bsz = batch_size * max(1, torch.cuda.device_count())
        dl = torch.utils.data.DataLoader(
            ImgDS(paths, processor),
            batch_size=effective_bsz,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=False,
            prefetch_factor=2,
            collate_fn=custom_collate_fn,
        )

        feat_chunks = []
        for batch in tqdm(dl, desc='Extracting Features'):
            if batch is None:
                continue
            batch_data, batch_files = batch
            feat_chunks.append(extract_features(batch_data, model, device=device))
            files.extend(batch_files)
        feats = np.concatenate(feat_chunks, axis=0).astype(np.float32)
        logger.info(f'Embedding matrix shape: {feats.shape}')

    index = build_faiss_index(feats, gpu=(not cpu_index))
    if not skip_gallery_vis and not debug:
        logger.info('Generating similarity visualization samples...')
        logger.info(f'Saving to: {root}')
        draw_gallery(index, feats, files, root)

    if debug:
        logger.info('Debug mode; skip grouping/removal.')
        return {"total": len(paths), "duplicates": 0, "groups": []}

    logger.info('Grouping duplicates (cosine)...')
    groups = group_duplicates_cosine(index, feats, files, similarity)

    dup_count = 0
    for gid, group in enumerate(tqdm(groups, desc='Processing Duplicates')):
        random.shuffle(group)
        to_remove = group[1:]  # keep 1
        for item in to_remove:
            if not os.path.exists(item):
                continue
            dup_count += 1
            if move_out:
                dst_dir = os.path.join(move_out, f'group_{gid}')
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(item, os.path.join(dst_dir, os.path.basename(item)))
            else:
                os.remove(item)

    logger.info(f"✓ Completed. Original images: {len(paths)}, dup images: {dup_count}")
    return {"total": len(paths), "duplicates": dup_count, "groups": groups}


# ---------------------------------------
# pHash pipeline
# ---------------------------------------
def _phash_similarity_to_distance(similarity: float) -> int:
    """Map similarity in [0,1] to Hamming distance on 64-bit pHash."""
    similarity = float(np.clip(similarity, 0.0, 1.0))
    return int(round((1.0 - similarity) * _PHASH_BITS))


def _dedup_phash_pipeline(
        root: str,
        distance: Optional[int] = 5,
        similarity: Optional[float] = None,
        debug: bool = False,
        move_out: Optional[str] = None,
        recursive: bool = True,
):
    """pHash dedup; supports distance or similarity (mapped to distance)."""
    if distance is None:
        if similarity is not None:
            distance = _phash_similarity_to_distance(similarity)
            logger.info(f"pHash mapping similarity={similarity:.4f} -> distance={distance}")

    handle = PHash()
    root = root.rstrip('/').rstrip('\\')
    totals = 0
    removed = 0
    groups_all: List[List[str]] = []

    def process_one_dir(dir_path: str):
        nonlocal totals, removed, groups_all
        encodings = handle.encode_images(image_dir=dir_path)
        totals += len(encodings)

        if debug:
            dups = handle.find_duplicates(encoding_map=encodings, max_distance_threshold=distance)
            # Convert pairwise list mapping to disjoint sets
            parent: Dict[str, str] = {}

            def find(x):
                parent.setdefault(x, x)
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            for src, lst in dups.items():
                for dst in lst:
                    union(src, dst)
            comp: Dict[str, List[str]] = {}
            for f in encodings.keys():
                r = find(f)
                comp.setdefault(r, []).append(os.path.join(dir_path, f))
            groups = [v for v in comp.values() if len(v) > 1]
            groups_all.extend(groups)

            if move_out:
                # Visualize a few examples into move_out for inspection
                os.makedirs(move_out, exist_ok=True)
                for g in groups[:10]:
                    try:
                        imgs = []
                        for fp in g[:6]:
                            arr = imageio.imread(fp)
                            arr = cv2.resize(arr, (128, 256))[:, :, :3]
                            imgs.append(arr)
                        vis = np.hstack(imgs)
                        io.imsave(os.path.join(move_out, str(uuid.uuid4()) + '.jpg'), vis)
                    except Exception:
                        continue
            return

        # non-debug: remove/move duplicates directly
        to_remove = handle.find_duplicates_to_remove(encoding_map=encodings, max_distance_threshold=distance)
        for fname in tqdm(to_remove, desc=f"pHash removing {os.path.basename(dir_path)}"):
            src = os.path.join(dir_path, fname)
            if not os.path.exists(src):
                continue
            if move_out:
                os.makedirs(move_out, exist_ok=True)
                shutil.move(src, os.path.join(move_out, fname))
            else:
                os.remove(src)
            removed += 1

    # Traverse
    if not recursive:
        process_one_dir(root)
    else:
        # apply to root and its subdirs independently (common pHash usage)
        has_sub = False
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if os.path.isdir(p):
                has_sub = True
                process_one_dir(p)
        if not has_sub:
            process_one_dir(root)

    logger.info(f"✓ Completed. Scanned: {totals} files, removed: {removed}")
    return {"total": totals, "duplicates": removed, "groups": groups_all}


# ---------------------------------------
# Unified public API
# ---------------------------------------
def dedup_images(
        root: str,
        method: str = "clip",  # 'clip' or 'phash'
        similarity: Optional[float] = 0.98,  # CLIP similarity (0..1), or mapped for pHash if distance absent
        distance: Optional[int] = 10,  # pHash Hamming distance
        # CLIP-specific
        model: str = 'google/siglip-base-patch16-224',
        batch_size: int = 256,
        cpu_index: bool = False,
        skip_gallery_vis: bool = False,
        # Common
        move_out: Optional[str] = None,
        debug: bool = False,
        recursive: bool = True,
):
    """
    Run image dedup using a single unified interface.

    Args:
        root: directory of images
        method: 'clip' or 'phash'
        similarity: float in [0,1]; primary for CLIP; for pHash it will be mapped to distance if distance is None
        distance: int; primary for pHash
        model: CLIP/SigLIP HF id if method='clip'
        batch_size: CLIP extraction batch size (per GPU *adjusted*)
        cpu_index: use CPU FAISS even if GPUs exist
        skip_gallery_vis: skip gallery png export (CLIP only, non-debug)
        move_out: if set, move duplicates there instead of deleting
        debug: limit samples & avoid deletion; creates sample galleries (CLIP) or sample stacks (pHash)
        recursive: pHash per-subdir recursion behavior

    Returns:
        dict with {"total": int, "duplicates": int, "groups": List[List[str]]}
    """
    method = method.lower().strip()
    if method not in {"clip", "phash"}:
        raise ValueError("method must be 'clip' or 'phash'")

    if method == "clip":
        sim = 0.98 if similarity is None else float(similarity)
        return _dedup_clip_pipeline(
            root=root,
            similarity=sim,
            model_name=model,
            batch_size=batch_size,
            cpu_index=cpu_index,
            debug=debug,
            move_out=move_out,
            skip_gallery_vis=skip_gallery_vis,
        )
    else:
        # pHash
        return _dedup_phash_pipeline(
            root=root,
            distance=distance,
            similarity=similarity,
            debug=debug,
            move_out=move_out,
            recursive=recursive,
        )


# ---------------------------------------
# Fire CLI
# ---------------------------------------
if __name__ == "__main__":
    fire.Fire({
        "dedup_images": dedup_images
    })
