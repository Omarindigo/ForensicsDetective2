import os
import io
import sys
import numpy as np
from PIL import Image

from utils import (
    ORIGINAL_DIRS,
    AUGMENTED_BASE_DIR,
    CLASS_LABELS,
    CLASS_SUBDIRS,
    ensure_dirs,
    get_project_root,
    list_supported_files,
    load_single_image
)


def apply_gaussian_noise(img):
    sigma = np.random.uniform(5, 20)
    arr = np.array(img, dtype=np.float64)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode='L')


def apply_jpeg_compression(img):
    quality = int(np.random.uniform(20, 80))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('L')


def apply_dpi_downsampling(img):
    target_dpi = np.random.choice([150, 72])
    scale = target_dpi / 300.0
    w, h = img.size
    small = img.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.LANCZOS
    )
    return small.resize((w, h), Image.LANCZOS)


def apply_random_cropping(img):
    w, h = img.size
    pct_left = np.random.uniform(0.01, 0.03)
    pct_right = np.random.uniform(0.01, 0.03)
    pct_top = np.random.uniform(0.01, 0.03)
    pct_bottom = np.random.uniform(0.01, 0.03)

    left = int(w * pct_left)
    right = w - int(w * pct_right)
    top = int(h * pct_top)
    bottom = h - int(h * pct_bottom)

    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((w, h), Image.LANCZOS)


def apply_bit_depth_reduction(img):
    arr = np.array(img, dtype=np.uint8)
    arr = (arr // 16) * 17
    return Image.fromarray(arr, mode='L')


AUGMENTATION_FUNCTIONS = {
    'gaussian_noise': apply_gaussian_noise,
    'jpeg_compression': apply_jpeg_compression,
    'dpi_downsampling': apply_dpi_downsampling,
    'random_cropping': apply_random_cropping,
    'bit_depth_reduction': apply_bit_depth_reduction,
}


def augment_dataset(repo_root='.'):
    np.random.seed(42)
    project_root = get_project_root(repo_root)
    total_created = 0

    for label, source_dir_name in ORIGINAL_DIRS.items():
        source_dir = os.path.join(project_root, source_dir_name)
        class_subdir = CLASS_SUBDIRS[label]

        if not os.path.isdir(source_dir):
            print(f"WARNING: source directory not found: {source_dir}")
            continue

        source_files = list_supported_files(source_dir)

        print(f"\n{CLASS_LABELS[label]}: processing {len(source_files)} files")

        for aug_name, aug_func in AUGMENTATION_FUNCTIONS.items():
            out_dir = os.path.join(project_root, AUGMENTED_BASE_DIR, aug_name, class_subdir)
            ensure_dirs(out_dir)

            created_here = 0

            for fname in source_files:
                full_path = os.path.join(source_dir, fname)
                img = load_single_image(full_path).convert('L')
                augmented = aug_func(img)

                stem = os.path.splitext(fname)[0]
                out_name = f"{stem}_{aug_name}.png"
                augmented.save(os.path.join(out_dir, out_name))
                total_created += 1
                created_here += 1

            print(f"  {aug_name}: {created_here} images")

    print(f"\nDone. Created {total_created} augmented images.")


if __name__ == '__main__':
    repo_root = '.'
    if len(sys.argv) > 1:
        repo_root = sys.argv[1]
    augment_dataset(repo_root=repo_root)