"""Detect and crop vehicle instances for FID-A."""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop YOLO-detected vehicles to square images for FID-A."
    )
    parser.add_argument(
        "--input-dir",
        "--input_dir",
        dest="input_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model",
        "--model-path",
        "--model_path",
        dest="model",
        default="yolo11x.pt",
        help="Ultralytics YOLO checkpoint path or model name.",
    )
    parser.add_argument(
        "--image-size",
        "--img_size",
        dest="image_size",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--device",
        help="Ultralytics device, for example cuda:0, 0, or cpu.",
    )
    return parser.parse_args()


def list_images(folder: Path):
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory not found: {folder}")
    paths = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise RuntimeError(f"No supported images found in: {folder}")
    return paths


def main():
    args = parse_args()
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory must be empty: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "ultralytics is not installed. Install evaluation/requirements.txt."
        ) from error

    model = YOLO(args.model)
    image_paths = list_images(args.input_dir)
    crop_count = 0
    for image_path in tqdm(image_paths, desc=f"Cropping {args.input_dir}"):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Cannot read image: {image_path}")
            continue

        predict_kwargs = {"verbose": False}
        if args.device:
            predict_kwargs["device"] = args.device
        results = model(str(image_path), **predict_kwargs)
        if not results or results[0].boxes is None:
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        names = model.names
        height, width = image.shape[:2]
        for detection_index, (box, class_id) in enumerate(zip(boxes, classes)):
            label = names[int(class_id)]
            if label not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            crop = cv2.resize(crop, (args.image_size, args.image_size))
            output_path = (
                args.output_dir
                / f"{image_path.stem}_{detection_index}_{label}.jpg"
            )
            if not cv2.imwrite(str(output_path), crop):
                raise RuntimeError(f"Failed to write crop: {output_path}")
            crop_count += 1

    if crop_count == 0:
        raise RuntimeError(
            f"YOLO found no {sorted(VEHICLE_CLASSES)} instances in {args.input_dir}"
        )
    print(f"FID-A crops: {crop_count}")
    print(f"Output:      {args.output_dir}")


if __name__ == "__main__":
    main()
