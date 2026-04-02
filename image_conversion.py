import os
import numpy as np
from PIL import Image


def pdf_bytes_to_image(pdf_path, target_size=(200, 200)):
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


def batch_convert(input_dir, output_dir, target_size=(200, 200)):
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    print(f"Converting {len(pdf_files)} PDFs from {input_dir}...")

    for fname in pdf_files:
        img = pdf_bytes_to_image(os.path.join(input_dir, fname), target_size)
        out_name = os.path.splitext(fname)[0] + '.png'
        img.save(os.path.join(output_dir, out_name))

    print(f"  Saved {len(pdf_files)} PNGs to {output_dir}")


if __name__ == '__main__':
    print("PNG images are already provided in the repository.")
    print("Use this module only if you need to re-convert from raw PDFs.")
