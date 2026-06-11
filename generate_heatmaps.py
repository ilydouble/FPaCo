import json
import cv2
import numpy as np
from pathlib import Path
import argparse

def generate_heatmap(image_path, roi_json_path, output_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return

    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    with open(roi_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    regions = data.get("roiAnalysis", {}).get("regions", [])

    for r in regions:
        x, y, bw, bh = r["bbox"]
        conf = r.get("confidence", 1.0)

        px = int(x * w)
        py = int(y * h)
        pw = int(bw * w)
        ph = int(bh * h)

        heatmap[py:py+ph, px:px+pw] += conf

    # Smooth heatmap
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

    # Normalize
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)

    # Save raw heatmap
    cv2.imwrite(str(output_path), heatmap)

    # Optional: overlay
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, colored, 0.4, 0)

    overlay_path = output_path.with_name(output_path.stem + "_overlay.png")
    cv2.imwrite(str(overlay_path), overlay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--roi-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    roi_dir = Path(args.roi_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in image_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    print(f"Found {len(images)} images")

    for img in images:
        roi_file = roi_dir / f"{img.stem}.json"
        if not roi_file.exists():
            continue

        out_file = output_dir / f"{img.stem}_heatmap.png"
        generate_heatmap(img, roi_file, out_file)


if __name__ == "__main__":
    main()
