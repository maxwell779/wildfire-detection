from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    package_root = Path(__file__).resolve().parents[1]
    default_input = package_root.parent / "03_classification_folder_labels"
    default_output = package_root / "generated" / "classification_augmented_dataset"
    default_preview = package_root / "previews" / "classification"

    parser = argparse.ArgumentParser(
        description="Create preview images or an offline augmented dataset for CNN/ResNet classification.",
    )
    parser.add_argument(
        "--mode",
        choices=("preview", "dataset"),
        default="preview",
        help="preview: generate sample comparison images, dataset: generate an offline augmented dataset",
    )
    parser.add_argument("--input-root", type=Path, default=default_input)
    parser.add_argument("--output-root", type=Path, default=default_output)
    parser.add_argument("--preview-root", type=Path, default=default_preview)
    parser.add_argument("--copies-per-image", type=int, default=1)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def iter_images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def apply_random_transform(image: Image.Image, rng: random.Random) -> tuple[Image.Image, list[str]]:
    aug = image.copy()
    operations: list[str] = []

    if rng.random() < 0.5:
        aug = aug.transpose(Image.FLIP_LEFT_RIGHT)
        operations.append("flip")

    brightness = rng.uniform(0.85, 1.2)
    if abs(brightness - 1.0) > 0.06:
        aug = ImageEnhance.Brightness(aug).enhance(brightness)
        operations.append(f"bright_{brightness:.2f}")

    contrast = rng.uniform(0.85, 1.15)
    if abs(contrast - 1.0) > 0.05:
        aug = ImageEnhance.Contrast(aug).enhance(contrast)
        operations.append(f"contrast_{contrast:.2f}")

    blur_radius = rng.uniform(0.0, 1.2)
    if blur_radius > 0.45:
        aug = aug.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        operations.append(f"blur_{blur_radius:.2f}")

    if not operations:
        operations.append("identity")

    return aug, operations


def build_side_by_side(original: Image.Image, augmented: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", (original.width + augmented.width, max(original.height, augmented.height)), "white")
    canvas.paste(original, (0, 0))
    canvas.paste(augmented, (original.width, 0))
    return canvas


def preview_samples(input_root: Path, preview_root: Path, preview_count: int, rng: random.Random) -> None:
    preview_root.mkdir(parents=True, exist_ok=True)

    samples: list[tuple[str, Path]] = []
    for category_dir in sorted((input_root / "train").iterdir()):
        if category_dir.is_dir():
            for image_path in iter_images(category_dir):
                samples.append((category_dir.name, image_path))

    rng.shuffle(samples)
    selected = samples[:preview_count]

    for index, (category, image_path) in enumerate(selected, start=1):
        with Image.open(image_path).convert("RGB") as image:
            augmented, operations = apply_random_transform(image, rng)
            comparison = build_side_by_side(image, augmented)
            op_suffix = "_".join(operations[:3])
            out_name = f"{index:02d}_{category}_{image_path.stem}_{op_suffix}.png"
            comparison.save(preview_root / out_name)


def copy_split_as_is(src_split: Path, dst_split: Path) -> None:
    shutil.copytree(src_split, dst_split)


def generate_dataset(
    input_root: Path,
    output_root: Path,
    copies_per_image: int,
    rng: random.Random,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")

    manifest_rows: list[dict[str, str]] = []
    output_root.mkdir(parents=True, exist_ok=True)

    for split in ("val", "test"):
        copy_split_as_is(input_root / split, output_root / split)

    train_src = input_root / "train"
    train_dst = output_root / "train"

    for category_dir in sorted(train_src.iterdir()):
        if not category_dir.is_dir():
            continue

        dst_category_dir = train_dst / category_dir.name
        dst_category_dir.mkdir(parents=True, exist_ok=True)

        for image_path in iter_images(category_dir):
            shutil.copy2(image_path, dst_category_dir / image_path.name)

            with Image.open(image_path).convert("RGB") as image:
                for copy_index in range(1, copies_per_image + 1):
                    augmented, operations = apply_random_transform(image, rng)
                    out_name = f"{image_path.stem}_aug{copy_index:02d}{image_path.suffix.lower()}"
                    out_path = dst_category_dir / out_name
                    augmented.save(out_path)
                    manifest_rows.append(
                        {
                            "split": "train",
                            "category": category_dir.name,
                            "original_image": str(image_path),
                            "augmented_image": str(out_path),
                            "operations": ",".join(operations),
                        }
                    )

    manifest_path = output_root / "augmentation_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["split", "category", "original_image", "augmented_image", "operations"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.mode == "preview":
        preview_samples(args.input_root, args.preview_root, args.preview_count, rng)
        print(f"Saved classification previews to: {args.preview_root}")
        return

    generate_dataset(args.input_root, args.output_root, args.copies_per_image, rng)
    print(f"Created classification augmented dataset at: {args.output_root}")


if __name__ == "__main__":
    main()
