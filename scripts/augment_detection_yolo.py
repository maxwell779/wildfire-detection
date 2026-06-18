from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {"0": "smoke", "1": "fire"}


def parse_args() -> argparse.Namespace:
    package_root = Path(__file__).resolve().parents[1]
    default_input = package_root.parent / "01_detection_yolo"
    default_output = package_root / "generated" / "detection_augmented_yolo"
    default_preview = package_root / "previews" / "detection"

    parser = argparse.ArgumentParser(
        description="Create preview images or an offline augmented YOLO dataset for detection models.",
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


def parse_label_file(label_path: Path) -> list[dict[str, float | str]]:
    boxes: list[dict[str, float | str]] = []
    if not label_path.exists():
        return boxes

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        class_id, x_center, y_center, width, height = line.split()
        boxes.append(
            {
                "class_id": class_id,
                "x_center": float(x_center),
                "y_center": float(y_center),
                "width": float(width),
                "height": float(height),
            }
        )
    return boxes


def serialize_boxes(boxes: list[dict[str, float | str]]) -> str:
    lines: list[str] = []
    for box in boxes:
        lines.append(
            "{} {:.16g} {:.16g} {:.16g} {:.16g}".format(
                box["class_id"],
                box["x_center"],
                box["y_center"],
                box["width"],
                box["height"],
            )
        )
    return ("\n".join(lines) + "\n") if lines else ""


def apply_random_transform(
    image: Image.Image,
    boxes: list[dict[str, float | str]],
    rng: random.Random,
) -> tuple[Image.Image, list[dict[str, float | str]], list[str]]:
    aug = image.copy()
    aug_boxes = [box.copy() for box in boxes]
    operations: list[str] = []

    if rng.random() < 0.5:
        aug = aug.transpose(Image.FLIP_LEFT_RIGHT)
        for box in aug_boxes:
            box["x_center"] = 1.0 - float(box["x_center"])
        operations.append("flip")

    brightness = rng.uniform(0.85, 1.2)
    if abs(brightness - 1.0) > 0.06:
        aug = ImageEnhance.Brightness(aug).enhance(brightness)
        operations.append(f"bright_{brightness:.2f}")

    contrast = rng.uniform(0.85, 1.15)
    if abs(contrast - 1.0) > 0.05:
        aug = ImageEnhance.Contrast(aug).enhance(contrast)
        operations.append(f"contrast_{contrast:.2f}")

    blur_radius = rng.uniform(0.0, 1.1)
    if blur_radius > 0.45:
        aug = aug.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        operations.append(f"blur_{blur_radius:.2f}")

    if not operations:
        operations.append("identity")

    return aug, aug_boxes, operations


def draw_boxes(image: Image.Image, boxes: list[dict[str, float | str]]) -> Image.Image:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    width, height = rendered.size

    for box in boxes:
        x_center = float(box["x_center"]) * width
        y_center = float(box["y_center"]) * height
        box_width = float(box["width"]) * width
        box_height = float(box["height"]) * height
        x1 = x_center - box_width / 2.0
        y1 = y_center - box_height / 2.0
        x2 = x_center + box_width / 2.0
        y2 = y_center + box_height / 2.0
        color = "#3a7bd5" if box["class_id"] == "0" else "#e34f4f"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1 + 4, y1 + 4), CLASS_NAMES[str(box["class_id"])], fill=color)

    return rendered


def build_side_by_side(original: Image.Image, augmented: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", (original.width + augmented.width, max(original.height, augmented.height)), "white")
    canvas.paste(original, (0, 0))
    canvas.paste(augmented, (original.width, 0))
    return canvas


def preview_samples(input_root: Path, preview_root: Path, preview_count: int, rng: random.Random) -> None:
    preview_root.mkdir(parents=True, exist_ok=True)

    image_dir = input_root / "data" / "train" / "images"
    label_dir = input_root / "data" / "train" / "labels"
    candidates: list[tuple[Path, Path]] = []

    for image_path in iter_images(image_dir):
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists() and label_path.read_text(encoding="utf-8").strip():
            candidates.append((image_path, label_path))

    rng.shuffle(candidates)
    selected = candidates[:preview_count]

    for index, (image_path, label_path) in enumerate(selected, start=1):
        boxes = parse_label_file(label_path)
        with Image.open(image_path).convert("RGB") as image:
            augmented, aug_boxes, operations = apply_random_transform(image, boxes, rng)
            original_boxed = draw_boxes(image, boxes)
            augmented_boxed = draw_boxes(augmented, aug_boxes)
            comparison = build_side_by_side(original_boxed, augmented_boxed)
            op_suffix = "_".join(operations[:3])
            out_name = f"{index:02d}_{image_path.stem}_{op_suffix}.png"
            comparison.save(preview_root / out_name)


def write_detection_yaml(output_root: Path) -> None:
    yaml_text = "\n".join(
        [
            f"path: {output_root.as_posix()}",
            "train: data/train/images",
            "val: data/val/images",
            "test: data/test/images",
            "",
            "names: ['smoke', 'fire']",
            "nc: 2",
            "",
        ]
    )
    (output_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def generate_dataset(
    input_root: Path,
    output_root: Path,
    copies_per_image: int,
    rng: random.Random,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")

    shutil.copytree(input_root / "data", output_root / "data")
    write_detection_yaml(output_root)

    train_image_dir = output_root / "data" / "train" / "images"
    train_label_dir = output_root / "data" / "train" / "labels"
    manifest_rows: list[dict[str, str]] = []

    for image_path in iter_images(train_image_dir):
        label_path = train_label_dir / f"{image_path.stem}.txt"
        boxes = parse_label_file(label_path)

        with Image.open(image_path).convert("RGB") as image:
            for copy_index in range(1, copies_per_image + 1):
                augmented, aug_boxes, operations = apply_random_transform(image, boxes, rng)
                aug_image_name = f"{image_path.stem}_aug{copy_index:02d}{image_path.suffix.lower()}"
                aug_label_name = f"{image_path.stem}_aug{copy_index:02d}.txt"
                aug_image_path = train_image_dir / aug_image_name
                aug_label_path = train_label_dir / aug_label_name
                augmented.save(aug_image_path)
                aug_label_path.write_text(serialize_boxes(aug_boxes), encoding="utf-8")
                manifest_rows.append(
                    {
                        "split": "train",
                        "original_image": str(image_path),
                        "original_label": str(label_path),
                        "augmented_image": str(aug_image_path),
                        "augmented_label": str(aug_label_path),
                        "operations": ",".join(operations),
                    }
                )

    manifest_path = output_root / "augmentation_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "split",
                "original_image",
                "original_label",
                "augmented_image",
                "augmented_label",
                "operations",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.mode == "preview":
        preview_samples(args.input_root, args.preview_root, args.preview_count, rng)
        print(f"Saved detection previews to: {args.preview_root}")
        return

    generate_dataset(args.input_root, args.output_root, args.copies_per_image, rng)
    print(f"Created detection augmented dataset at: {args.output_root}")


if __name__ == "__main__":
    main()
