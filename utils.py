import os
import numpy as np
from PIL import Image

TARGET_SIZE = (200, 200)
RANDOM_STATE = 42
TEST_SIZE = 0.2

CLASS_NAMES = ['Word', 'Google Docs', 'Python']
CLASS_LABELS = {0: 'Word', 1: 'Google Docs', 2: 'Python'}

CLASS_SUBDIRS = {
    0: 'word',
    1: 'google',
    2: 'python',
}

ORIGINAL_DIRS = {
    0: 'word_pdfs',
    1: 'google_docs_pdfs',
    2: 'python_pdfs',
}

AUGMENTATION_TYPES = [
    'gaussian_noise',
    'jpeg_compression',
    'dpi_downsampling',
    'random_cropping',
    'bit_depth_reduction',
]

AUGMENTED_BASE_DIR = os.path.join('data', 'augmented_images')

RESULTS_DIR = 'results'
CONFUSION_DIR = os.path.join(RESULTS_DIR, 'confusion_matrices')
ROBUSTNESS_DIR = os.path.join(RESULTS_DIR, 'robustness_plots')


def get_project_root(repo_root='.'):
    return os.path.abspath(repo_root)


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def pdf_bytes_to_image(pdf_path, target_size=TARGET_SIZE):
    with open(pdf_path, 'rb') as f:
        raw = f.read()

    byte_array = np.frombuffer(raw, dtype=np.uint8)

    side = int(np.ceil(np.sqrt(len(byte_array))))
    padded = np.zeros(side * side, dtype=np.uint8)
    padded[:len(byte_array)] = byte_array
    img_array = padded.reshape((side, side))

    img = Image.fromarray(img_array, mode='L')
    img = img.resize(target_size, Image.LANCZOS)
    return img


def load_single_image(filepath, target_size=TARGET_SIZE):
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.png':
        img = Image.open(filepath).convert('L')
        img = img.resize(target_size, Image.LANCZOS)
        return img

    if ext == '.pdf':
        return pdf_bytes_to_image(filepath, target_size=target_size)

    raise ValueError(f"Unsupported file type: {filepath}")


def list_supported_files(directory):
    if not os.path.isdir(directory):
        return []
    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith('.png') or f.lower().endswith('.pdf')
    ])


def load_images_from_dir(directory, label, max_samples=None,
                         target_size=TARGET_SIZE):
    X_list, y_list, filenames = [], [], []

    if not os.path.isdir(directory):
        print(f"  WARNING: directory not found: {directory}")
        return X_list, y_list, filenames

    files = list_supported_files(directory)

    if max_samples is not None:
        files = files[:max_samples]

    for fname in files:
        try:
            img = load_single_image(os.path.join(directory, fname), target_size=target_size)
            X_list.append(np.array(img).flatten())
            y_list.append(label)
            filenames.append(fname)
        except Exception as e:
            print(f"  Error loading {fname}: {e}")

    return X_list, y_list, filenames


def load_original_dataset(repo_root='.', max_samples_per_class=None):
    print("Loading original 3-class dataset...")
    project_root = get_project_root(repo_root)

    all_X, all_y, all_fnames = [], [], []

    for label, dirname in ORIGINAL_DIRS.items():
        dirpath = os.path.join(project_root, dirname)
        X_part, y_part, f_part = load_images_from_dir(
            dirpath, label, max_samples=max_samples_per_class
        )
        all_X.extend(X_part)
        all_y.extend(y_part)
        all_fnames.extend(f_part)
        print(f"  {CLASS_LABELS[label]}: loaded {len(X_part)} files from {dirpath}")

    X = np.array(all_X)
    y = np.array(all_y)

    if len(X) == 0:
        raise FileNotFoundError(
            "No source files were found.\n"
            "Make sure these folders exist and contain .png or .pdf files:\n"
            "  - word_pdfs/\n"
            "  - google_docs_pdfs/\n"
            "  - python_pdfs/\n"
            f"Current project root: {project_root}"
        )

    print(f"Total: {len(X)} samples, {X.shape[1]} features each")
    return X, y, all_fnames


def load_augmented_dataset(augmentation_type, repo_root='.',
                           max_samples_per_class=None):
    project_root = get_project_root(repo_root)

    all_X, all_y = [], []

    for label, subdir in CLASS_SUBDIRS.items():
        dirpath = os.path.join(project_root, AUGMENTED_BASE_DIR,
                               augmentation_type, subdir)
        X_part, y_part, _ = load_images_from_dir(
            dirpath, label, max_samples=max_samples_per_class
        )
        all_X.extend(X_part)
        all_y.extend(y_part)

    return np.array(all_X), np.array(all_y)