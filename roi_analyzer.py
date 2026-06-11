#!/usr/bin/env python3
"""
ROI Analysis Module - Using Gemini API to generate regions of interest
for MIAS / APTOS / OCTA datasets
"""

import base64
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)


class GeminiROIAnalyzer:
    """Use Gemini API to generate ROI annotations for medical images"""

    def __init__(self, api_key: str, base_url: str = "https://yunwu.ai/v1"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    @staticmethod
    def encode_image(image_path: Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def get_roi_prompt(dataset: str) -> str:
        dataset = dataset.lower()

        if dataset == "mias":
            return """
You are a medical imaging expert analyzing mammography images (MIAS dataset).

Task:
Identify clinically significant lesion regions used for heatmap generation.

Output ONLY a valid JSON object:

{
  "roiAnalysis": {
    "regions": [
      {
        "regionType": "Mass | Calcification | Distortion | Asymmetry",
        "bbox": [x, y, width, height],
        "confidence": 0.0-1.0
      }
    ]
  }
}

All bounding box values MUST be normalized between 0 and 1.
"""

        if dataset == "aptos":
            return """
You are analyzing retinal fundus images from the APTOS dataset.

Identify diabetic retinopathy related regions:
- Microaneurysms
- Hemorrhages
- Exudates
- Optic Disc

Output ONLY a valid JSON object with normalized bounding boxes.
"""

        if dataset == "octa":
            return """
You are analyzing OCTA images.

Identify vascular-related regions of interest:
- FAZ (Foveal Avascular Zone)
- Vessel dropout
- Capillary non-perfusion

Output ONLY a valid JSON object with normalized bounding boxes.
"""
        if dataset == "fingerprint":
            return """
You are analyzing fingerprint images.

Identify regions with high ridge density and distinctive minutiae
for heatmap generation.

Output ONLY a valid JSON object with normalized bounding boxes.
"""


        raise ValueError("Unsupported dataset")

    def analyze_image(self, image_path: Path, dataset: str) -> Optional[Dict[str, Any]]:
        try:
            base64_image = self.encode_image(image_path)
            prompt = self.get_roi_prompt(dataset)

            logger.info("Calling Gemini API...")
            response = self.client.chat.completions.create(
                model="gemini-2.5-pro",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ],
                    }
                ],
                temperature=0.1,
            )

            content = response.choices[0].message.content.strip()

            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "").strip()

            return json.loads(content)

        except Exception as e:
            logger.error(f"Analysis failed for {image_path.name}: {e}")
            return None

    def analyze_batch(self, image_dir: Path, output_dir: Path, dataset: str):
        image_dir = Path(image_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images = [p for p in image_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

        logger.info(f"Found {len(images)} images")

        for idx, img in enumerate(images, 1):
            logger.info(f"[{idx}/{len(images)}] Processing {img.name}")

            output_file = output_dir / f"{img.stem}.json"
            if output_file.exists():
                logger.info("⏭️ Skipped (already exists)")
                continue

            result = self.analyze_image(img, dataset)
            if result:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                logger.info(f"✅ Saved: {output_file}")
            else:
                logger.error("❌ Failed")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    import argparse

    parser = argparse.ArgumentParser(description="Gemini ROI Generator")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True, choices=["mias", "aptos", "octa"])
    args = parser.parse_args()

    api_key = os.getenv("YUNWU_API_KEY")
    if not api_key:
        logger.error("YUNWU_API_KEY not set")
        return

    analyzer = GeminiROIAnalyzer(api_key)
    analyzer.analyze_batch(
        image_dir=Path(args.image_dir),
        output_dir=Path(args.output_dir),
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
