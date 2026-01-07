#!/usr/bin/env python3
"""
指纹分析模块 - 使用Gemini API进行结构化分析
"""

import base64
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os



logger = logging.getLogger(__name__)


class FingerprintGeminiAnalyzer:
    """使用Gemini API分析指纹图像并返回结构化JSON数据"""
    
    def __init__(self, api_key: str, base_url: str = "https://yunwu.ai/v1"):
        """
        初始化分析器
        
        Args:
            api_key: 云雾AI API密钥
            base_url: API基础URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    @staticmethod
    def encode_image(image_path: Path) -> str:
        """将图像编码为base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    @staticmethod
    def get_analysis_prompt() -> str:
        """获取指纹分析提示词"""
        return """**Role:**
You are an expert forensic analyst specializing in dermatoglyphics (the scientific study of fingerprints).
Your task is to meticulously analyze the provided fingerprint image and output a structured JSON object.

**CRITICAL: L vs W Classification Rules**

This is the most important distinction. Many errors occur here. Follow these rules STRICTLY:

1. **LOOP (L) - 箕形纹:**
   - **MUST HAVE:** Exactly 1 Delta (not 2, not 0)
   - **Core Feature:** Ridges enter from ONE side, curve back, and exit from the SAME side
   - **Visual Pattern:** Forms a "U" shape or "Ω" shape
   - **Ridge Flow:** Unidirectional - all ridges flow in one general direction
   - **Key Test:** If you can trace a path from one side to the other without crossing a ridge, it's a LOOP

   **Sub-types (SHAPE AND FLOW DIRECTION ARE KEY):**

   - `Lu (Ulnar Loop - 水纹)`:
     - **Visual pattern:** HORSESHOE shape (马蹄形)
     - **Ridge flow:** Overall ridges form a U-shape flowing OUT towards the LITTLE FINGER (小指)
     - **Key feature:** U-shaped ridges open and flow towards the pinky side
     - **Delta:** Has 1 outer delta (外三叉点)
     - **Visual test:** If you see a horseshoe/U-shape opening towards the little finger, it's Lu

   - `Lr (Radial Loop - 火纹)`:
     - **Visual pattern:** HORSESHOE shape (马蹄形)
     - **Ridge flow:** Overall ridges form a U-shape flowing OUT towards the THUMB (大拇指)
     - **Key feature:** U-shaped ridges open and flow towards the thumb side
     - **Delta:** Has 1 outer delta (外三叉点)
     - **Visual test:** If you see a horseshoe/U-shape opening towards the thumb, it's Lr

   - `Lf (Falling Loop - 下垂纹)`:
     - **Visual pattern:** U-shape flowing out with CONVERGENCE at the top
     - **Ridge flow:** Overall ridges form a U-shape flowing out
     - **Key feature:** Has 1 outer delta (外三叉点) + CONVERGENCE POINT at the top of U
     - **CRITICAL:** At the TOP of the U-shape, ≥4 ridges CONVERGE forming an ACUTE ANGLE and DO NOT flow out
     - **Convergence test:** The converging ridges form a sharp angle/wedge and STOP (不流出)
     - **Visual test:** If you see U-shape + ≥4 ridges converging at top forming acute angle, it's Lf

2. **WHORL (W) - 螺纹:**
   - **MUST HAVE:** 2 or more Deltas (not 1, not 0)
   - **Core Feature:** Ridges form complete, closed-circuit patterns around a central core
   - **Visual Pattern:** Forms concentric circles, spirals, or intertwined loops
   - **Ridge Flow:** Circular or spiral - ridges revolve around the center
   - **Key Test:** If ridges completely surround the core and you cannot trace a simple path through, it's a WHORL

   **Sub-types (CRITICAL: COUNT INNER 3 CIRCLES AND MEASURE ASPECT RATIO):**

   - `Wt (Concentric Whorl - 同心圆纹)`:
     - **Visual pattern:** Like a TARGET/BULLSEYE (打靶靶子)
     - **Central ridges:** Form CONCENTRIC CIRCLES around the center
     - **CRITICAL TEST (within inner 3 circles):** Within the innermost 3 circles (包含第三圈), there must be AT LEAST ONE complete CLOSED and INDEPENDENT loop/ring
       * OR the ridge flow within 3 circles is independent with a CLOSING TENDENCY (有封闭的趋势，几乎要封闭上)
     - **Outer ridges:** Show concentric circles OR spiral flow
     - **Visual test:** If you see a target-like pattern with a closed/nearly-closed ring within 3 circles, it's Wt

   - `Ws (Spiral Whorl - 螺旋纹)`:
     - **Visual pattern:** SPIRAL shape (螺旋状)
     - **Central ridges:** Form SPIRAL pattern around the center
     - **CRITICAL MEASUREMENT:** Within the central 3 circles, the LENGTH is LESS THAN 2× the WIDTH (长是宽的2倍以内，不包含2倍)
       * Aspect ratio of inner 3 circles: length/width < 2.0
     - **Outer ridges:** Flow out in SPIRAL pattern
     - **Visual test:** If you see spiral pattern with aspect ratio < 2.0 in the center, it's Ws

   - `We (Press Whorl - 压纹)`:
     - **Visual pattern:** FOOTBALL/OLIVE shape (橄榄球状)
     - **Central ridges:** Form SPIRAL pattern around the center (similar to Ws)
     - **CRITICAL MEASUREMENT:** Within the central 3 circles, the LENGTH is ≥ 2.5× the WIDTH (长是宽的2.5倍及以上)
       * Aspect ratio of inner 3 circles: length/width ≥ 2.5
     - **Outer ridges:** Flow out in SPIRAL pattern
     - **Visual test:** If you see elongated spiral with aspect ratio ≥ 2.5 in the center, it's We

   - `Wc (Composite Whorl - 复合纹)`:
     - **Visual pattern:** TAI-CHI shape (太极形状)
     - **Central ridges:** Form a complete S-LINE (完整S线)
     - **CRITICAL S-LINE TEST:**
       * Each side of the S-line contains ≥2 ridges (S线两侧内各含两条及以上纹脊线)
       * Left center line flows RIGHT into a delta (左边中心线往右流进三叉点)
       * Right center line flows LEFT into a delta (右边中心线往左流进三叉点)
       * Note: At least ONE center line flowing into delta is sufficient (中心线有一条流进也算)
     - **Visual test:** If you see S-line with ridges on both sides flowing INTO deltas, it's Wc

   - `Wd (Double Loop Whorl - 双箕斗纹)`:
     - **Visual pattern:** TAI-CHI shape but FLATTER (太极形状，形状较扁)
     - **Central ridges:** Form a complete S-LINE (完整S线)
     - **CRITICAL S-LINE TEST:**
       * Each side of the S-line contains ≥2 ridges (S线两侧内各含两条及以上纹脊线)
       * Left center line flows RIGHT OUT OF a delta (左边中心线往右流出三叉点)
       * Right center line flows LEFT OUT OF a delta (右边中心线往左流出三叉点)
     - **Distinction from Wc:** Wc has center lines flowing INTO deltas, Wd has center lines flowing OUT OF deltas
     - **Visual test:** If you see flattened S-line with ridges flowing OUT OF deltas, it's Wd

   - `Wp (Peacock Whorl - 孔雀翎纹)`:
     - **Visual pattern:** PEACOCK FEATHER shape (孔雀翎形状)
     - **Delta structure:** Has 1 INNER delta (内三叉) + 1 OUTER delta (外三叉)
     - **Ridge flow:** Ridges on both sides show SYMMETRICAL or CROSSING flow converging together
     - **Key feature:** Two-sided ridges converge symmetrically or cross each other
     - **Visual test:** If you see peacock feather with inner + outer delta and symmetrical convergence, it's Wp

   - `WrI (Incomplete Peacock - 未完整孔雀纹)`:
     - **Visual pattern:** INCOMPLETE PEACOCK (没长成的孔雀)
     - **Delta structure:** Has 1 INNER delta (含不完整内三叉) + 1 OUTER delta (外三叉)
     - **Key feature:** The inner delta may be INCOMPLETE (不完整内三叉)
     - **Distinction from Wp:** Wp has complete inner delta, WrI has incomplete inner delta
     - **Visual test:** If you see peacock-like pattern with incomplete inner delta, it's WrI

3. **ARCH (A) - 弓形纹:**
   - **MUST HAVE:** 0 Deltas
   - **Core Feature:** Ridges enter from one side and flow to the other with gentle rise in middle
   - **Visual Pattern:** Wave-like or tent-like rise
   - **Ridge Flow:** Simple, unidirectional flow

   **Sub-types (SHAPE AND STRUCTURE ARE KEY):**

   - `Aul (Ulnar Loop Arch - 地纹)`:
     - **Visual pattern:** HORSESHOE shape (马蹄形)
     - **Ridge flow:** Overall ridges form a U-shape flowing towards the LITTLE FINGER (小拇指)
     - **Key feature:** ARCH characteristics dominate (弧的特征占主体)
     - **CRITICAL:** Has 0 deltas (this distinguishes it from Lu which has 1 delta)
     - **Visual test:** If you see horseshoe/U-shape towards pinky BUT with 0 deltas, it's Aul
     - **Distinction from Lu:** Lu has 1 delta, Aul has 0 deltas

   - `As (Simple Arch - 土纹)`:
     - **Visual pattern:** MOUND or BOW shape, relatively FLAT (土丘、弓型，形状较扁)
     - **Ridge flow:** Smooth arched ridges from one side to the other
     - **Key feature:** Simple, flat arch with no complexity
     - **Visual test:** If you see a flat, simple bow/mound shape, it's As

   - `At (Tented Arch - 帐篷纹)`:
     - **Visual pattern:** ARCHED ridges rising HIGH, with a "人" shape in the middle like a TENT (弧线高高隆起，中间像一个"人"字，好似帐篷)
     - **Ridge flow:** Ridges rise sharply in the center forming a tent-like peak
     - **Key feature:** High arch with a sharp "人" (person) character shape in the center
     - **Visual test:** If you see high-rising arch with "人" shape in center, it's At

   - `Ae (Elevated Arch - 突起山丘纹)`:
     - **Visual pattern:** RAISED HILL shape (突起山丘)
     - **Ridge structure:** Bottom ridges form an ARC that encloses a TRIANGULAR or CIRCULAR shape
       * OR contains an independent small WHITE BLOCK (独立的小白块)
     - **Key feature:** Arc at the bottom forms enclosed triangular/circular area or has white block
     - **Visual test:** If you see raised hill with enclosed triangle/circle at bottom or white block, it's Ae

4. **X - Variant/Composite Family (变异/复合纹):**
   - **Core Feature:** Multiple pattern types appear SIMULTANEOUSLY on the SAME FINGER (多种纹型同时出现在同一手指上)
   - **Examples:**
     * Part of the finger shows Loop pattern, another part shows Whorl pattern
     * Mixed characteristics that don't fit cleanly into A, L, or W
   - **Visual test:** If you see multiple distinct pattern types on one finger, classify as X

**COMMON MISTAKES TO AVOID:**

❌ MISTAKE 1: Confusing Loop with Whorl
   - A Loop with a prominent curve might LOOK like it has 2 deltas
   - But a true Delta is where ridges DIVERGE/SPLIT, not just curve
   - Count carefully: are there really 2 distinct divergence points?
   - If unsure, it's probably a LOOP (delta=1)

❌ MISTAKE 2: Counting ridge bifurcations as deltas
   - A bifurcation is where ONE ridge splits into TWO
   - A delta is where MULTIPLE ridges diverge from a point
   - These are different!

❌ MISTAKE 3: Misidentifying the core
   - The core is the innermost ridge formation
   - For Loops: it's the U-shaped or Ω-shaped part
   - For Whorls: it's the central circular/spiral part
   - Look carefully at the CENTER of the pattern

❌ MISTAKE 4: Confusing Aul (Arch) with Lu (Loop)
   - Both have horseshoe/U-shape flowing towards little finger
   - But Aul has 0 deltas (Arch family), Lu has 1 delta (Loop family)
   - Always count deltas carefully to distinguish

❌ MISTAKE 5: Confusing Ws with We
   - Both have spiral pattern in center
   - Measure the aspect ratio of the inner 3 circles:
     * Ws: length/width < 2.0
     * We: length/width ≥ 2.5
   - Use the aspect ratio measurement to distinguish

❌ MISTAKE 6: Confusing Wc with Wd
   - Both have S-line with ridges on both sides
   - Check the flow direction of center lines:
     * Wc: Center lines flow INTO deltas (流进三叉点)
     * Wd: Center lines flow OUT OF deltas (流出三叉点)
   - Look carefully at the direction of flow

❌ MISTAKE 7: Confusing Wp with WrI
   - Both have peacock-like pattern with inner + outer delta
   - Check the completeness of inner delta:
     * Wp: Inner delta is COMPLETE
     * WrI: Inner delta is INCOMPLETE (不完整内三叉)
   - Examine the inner delta structure carefully

**DECISION TREE:**

1. Count the number of DELTAS (divergence points):
   - 0 deltas → ARCH (A)
   - 1 delta → LOOP (L)
   - 2+ deltas → WHORL (W)

2. If uncertain about delta count:
   - Look at the ridge flow direction
   - Loops have unidirectional flow
   - Whorls have circular/spiral flow
   - Use ridge flow as secondary confirmation

3. If still uncertain:
   - Set confidence < 0.7
   - Mark as "Uncertain"
   - Don't force a classification

**CONFIDENCE GUIDELINES:**

- High confidence (0.8-1.0): Clear delta count, obvious ridge pattern
- Medium confidence (0.6-0.8): Reasonable delta count, some ambiguity
- Low confidence (0.4-0.6): Unclear deltas, ambiguous pattern
- Very low confidence (<0.4): Cannot reliably classify, mark as "Uncertain"

**CRITICAL REQUIREMENTS (MUST FOLLOW):**

These requirements address common quality issues. Failure to follow these will result in incomplete analysis:

1. **notableFeatures (MANDATORY):**
   - You MUST provide at least 3 notable features in ridgeCharacteristics.notableFeatures
   - If the image quality is poor and features are unclear, still provide 3 entries with type="unclear" and explain why
   - Acceptable feature types: bifurcation, ridge ending, dot, enclosure, delta, core, s-line, inner delta, convergence point, etc.
   - Each feature MUST have: type, location, description

2. **Loop Opening Direction (MANDATORY for Lr/Lu):**
   - For ALL Loop sub-types (Lr, Lu, Lf, Lrf), you MUST explicitly state the opening direction
   - In the "reasoning" field of patternSubType, you MUST include phrases like:
     * "opening facing the RADIAL side (thumb side)" for Lr
     * "opening facing the ULNAR side (pinky side)" for Lu/Lf
   - In ridgeCharacteristics.flowDirection, you MUST specify: "Towards thumb (Lr)" or "Towards little finger (Lu/Lf)"
   - If you cannot determine the direction, set confidence < 0.7 and explain why

3. **Whorl Inner Delta Completeness (MANDATORY for Wp/WrI):**
   - For Whorl sub-types Wp and WrI, you MUST check the inner delta completeness
   - Add an entry in notableFeatures with type="inner delta"
   - In the description, explicitly state: "COMPLETE" or "INCOMPLETE"
   - In structuralFeatures.delta.innerDeltaComplete, set true (Wp) or false (WrI)
   - In patternSubType.reasoning, mention: "inner delta is complete/incomplete"

4. **Arch Sharpness Description (MANDATORY for At/As):**
   - For Arch sub-types At and As, you MUST describe the sharpness
   - At (Tented Arch): Use words like "SHARP", "POINTED", "steep angle", "tent-like peak"
   - As (Simple Arch): Use words like "SMOOTH", "GENTLE", "gradual curve", "wave-like"
   - Include this description in both:
     * structuralFeatures.core.description
     * patternSubType.reasoning

5. **Whorl Shape and Aspect Ratio (MANDATORY for Wt/Ws/We):**
   - For Whorl sub-types Wt, Ws, We, you MUST describe the shape and aspect ratio
   - In structuralFeatures.core.description, include:
     * Wt: "CIRCULAR" or "aspect ratio close to 1:1", mention "concentric circles"
     * Ws: "SPIRAL" with "aspect ratio 1.2-2.0", mention rotation direction
     * We: "ELONGATED" or "elliptical" with "aspect ratio > 2.0"
   - In structuralFeatures.core.aspectRatioInner3Circles, provide estimated ratio (e.g., 1.5, 2.3)
   - Mention shape keywords: circular, spiral, elongated, elliptical, concentric

6. **Whorl S-Line Check (MANDATORY for Wc/Wd):**
   - For Whorl sub-types Wc and Wd, you MUST check for S-line presence
   - Add an entry in notableFeatures with type="s-line"
   - Describe the flow direction:
     * Wc: "S-line flows INTO the deltas"
     * Wd: "S-line flows OUT from the deltas"
   - In structuralFeatures.sLine.centerLineFlowDirection, specify: "Into deltas" or "Out of deltas"
   - If no S-line is visible, state: "No clear S-line visible" and set hasCompleteSLine to false

**Instructions:**

Analyze the provided fingerprint image step-by-step:

1. **Delta Count (MOST IMPORTANT):**
   - Count the number of divergence points where ridges split
   - Be very careful and explicit about this count
   - State your reasoning for the count

2. **Ridge Flow Pattern:**
   - Describe the overall direction of ridge flow
   - Is it unidirectional (Loop) or circular/spiral (Whorl)?

3. **Overall Classification:**
   - Determine the main family (W, L, A, or X)
   - State your confidence level (0.0 to 1.0)
   - Provide reasoning based on delta count and ridge flow

4. **Sub-type Identification (EXTRA DETAILED):**

   **FOR LOOP - ANALYZE U-SHAPE DIRECTION AND CONVERGENCE:**
   - If classified as LOOP, determine the U-shape opening direction:
     * Does the U-shape open towards the THUMB? → Lr (火纹)
     * Does the U-shape open towards the LITTLE FINGER? → Lu (水纹) or Lf (下垂纹)
   - For Loop sub-types:
     * Lr: Horseshoe/U-shape opening towards thumb
     * Lu: Horseshoe/U-shape opening towards little finger, NO convergence at top
     * Lf: U-shape opening out + ≥4 ridges CONVERGING at TOP forming acute angle (ridges DO NOT flow out)
   - **CRITICAL: Lu vs Lf distinction:**
     * Lu: U-shape towards pinky, smooth flow, no sharp convergence
     * Lf: U-shape + convergence point at top where ≥4 ridges meet and STOP
   - State your confidence and reasoning

   **FOR ARCH - ANALYZE SHAPE AND STRUCTURE:**
   - If classified as ARCH, analyze the shape carefully:
     * Is it a HORSESHOE/U-shape towards little finger? → Aul (地纹)
     * Is it a FLAT BOW/MOUND shape? → As (土纹)
     * Is it a HIGH-RISING arch with "人" shape in center? → At (帐篷纹)
     * Is it a RAISED HILL with enclosed triangle/circle or white block at bottom? → Ae (突起山丘纹)
   - For Arch sub-types:
     * Aul: Horseshoe/U-shape towards pinky, arch characteristics dominate, 0 deltas
     * As: Flat bow/mound shape, simple and flat
     * At: High-rising arch with "人" (person) shape in center like a tent
     * Ae: Raised hill with arc enclosing triangle/circle at bottom or white block
   - State your confidence and reasoning

   **FOR WHORL - ANALYZE CENTRAL REGION AND MEASURE ASPECT RATIO:**
   - If classified as WHORL, perform these steps:

   **STEP 1: Identify the overall pattern shape**
     * TARGET/BULLSEYE shape? → Likely Wt
     * SPIRAL shape? → Likely Ws or We (need to measure)
     * TAI-CHI/S-LINE shape? → Likely Wc or Wd (check flow direction)
     * PEACOCK FEATHER shape? → Likely Wp or WrI (check inner delta)

   **STEP 2: For Wt - Check for closed loop within 3 circles**
     * Count inward from the center: 1st circle, 2nd circle, 3rd circle
     * Within these 3 circles, is there at least ONE complete CLOSED and INDEPENDENT loop?
     * OR do the ridges show independent flow with CLOSING TENDENCY (almost closed)?
     * If YES → Wt

   **STEP 3: For Ws/We - Measure aspect ratio of inner 3 circles**
     * Identify the innermost 3 circles
     * Measure the LENGTH (longest dimension) and WIDTH (shortest dimension)
     * Calculate: aspect_ratio = length / width
     * If aspect_ratio < 2.0 → Ws (螺旋纹)
     * If aspect_ratio ≥ 2.5 → We (压纹)

   **STEP 4: For Wc/Wd - Check S-line flow direction**
     * Identify the complete S-line in the center
     * Check: Each side of S-line has ≥2 ridges?
     * Check flow direction of center lines:
       - Do center lines flow INTO deltas? → Wc (复合纹)
       - Do center lines flow OUT OF deltas? → Wd (双箕斗纹)

   **STEP 5: For Wp/WrI - Check inner delta completeness**
     * Identify inner delta and outer delta
     * Is the inner delta COMPLETE? → Wp (孔雀翎纹)
     * Is the inner delta INCOMPLETE? → WrI (未完整孔雀纹)

   - State your confidence and reasoning with specific measurements

5. **Structural Analysis:**
   - Count cores and deltas
   - Describe the core shape and pattern
   - **For Whorl:** Explicitly state core aspect ratio (length/width)
   - **For Whorl:** Describe whether pattern is concentric or spiral

6. **Ridge Analysis:**
   - Describe overall ridge flow
   - Note any clear minutiae (bifurcations, ridge endings)
   - **For Whorl:** Describe the rotation/spiral direction (clockwise/counterclockwise)

7. **Quality Assessment:**
   - Evaluate image quality
   - Note any issues (smudges, partial prints, etc.)

**Output Format:**
You MUST format your entire response as a single, valid JSON object.
Do not include any explanatory text before or after the JSON block.

{
  "fingerprintAnalysis": {
    "imageId": "Provide the original image identifier here if available, otherwise null",
    "analysisTimestamp": "Provide the current ISO 8601 timestamp",
    "familyClassification": {
      "predictedFamily": "String (Whorl, Loop, Arch, Variant, Uncertain)",
      "confidence": "Float (0.0-1.0)",
      "reasoning": "String - MUST include delta count and ridge flow reasoning"
    },
    "patternSubType": {
      "predictedSubTypeCode": "String (e.g., Wt, Lu, As, etc.)",
      "predictedSubTypeName": "String (e.g., Concentric Whorl, Ulnar Loop, etc.)",
      "confidence": "Float (0.0-1.0)",
      "reasoning": "String - MUST be at least 30 characters. For Loop: include opening direction (radial/ulnar). For Arch: include sharpness. For Whorl: include shape/aspect ratio/S-line/inner delta as applicable"
    },
    "structuralFeatures": {
      "core": {
        "count": "Integer",
        "type": "String (e.g., Dot, U-Shaped Loop, Tented Arch Apex, Circular, Spiral, S-Line, None)",
        "description": "String - MUST be at least 20 characters. For Whorl: include shape and aspect ratio. For Arch: include sharpness (sharp/smooth/gentle)",
        "centralRegionShape": "String or null - For Whorl: 'Target/Bullseye', 'Spiral', 'Football/Olive', 'Tai-Chi', 'Peacock Feather', or null",
        "aspectRatioInner3Circles": "Float or null - For Whorl Ws/We: length/width ratio of innermost 3 circles (Ws: <2.0, We: ≥2.5). REQUIRED for Wt/Ws/We",
        "hasClosedLoopWithin3Circles": "Boolean or null - For Whorl Wt: Does it have at least one closed/nearly-closed loop within 3 circles?",
        "pattern": "String or null - For Whorl: 'Concentric', 'Spiral', 'S-Line', 'Peacock', or null"
      },
      "delta": {
        "count": "Integer - CRITICAL: Must match family classification (A:0, L:1, W:2+)",
        "positionDescription": "String - Describe each delta location explicitly (MUST be at least 10 characters)",
        "hasInnerDelta": "Boolean or null - For Whorl Wp/WrI: Does it have an inner delta? REQUIRED for Wp/WrI",
        "hasOuterDelta": "Boolean or null - For Whorl Wp/WrI: Does it have an outer delta? REQUIRED for Wp/WrI",
        "innerDeltaComplete": "Boolean or null - For Whorl Wp/WrI: Is the inner delta complete? (Wp: true, WrI: false) REQUIRED for Wp/WrI"
      },
      "sLine": {
        "hasCompleteSLine": "Boolean or null - For Whorl Wc/Wd: Does it have a complete S-line? REQUIRED for Wc/Wd",
        "ridgesOnEachSide": "Integer or null - For Whorl Wc/Wd: Number of ridges on each side of S-line (should be ≥2) REQUIRED for Wc/Wd",
        "centerLineFlowDirection": "String or null - For Whorl Wc/Wd: 'Into deltas' (Wc) or 'Out of deltas' (Wd) REQUIRED for Wc/Wd"
      }
    },
    "ridgeCharacteristics": {
      "overallPattern": "String - Describe ridge flow direction and shape (MUST be at least 30 characters)",
      "flowDirection": "String or null - For Loop: 'Towards thumb (Lr)', 'Towards little finger (Lu/Lf)', or null",
      "ridgeCount": "Integer or null - Count visible ridges",
      "hasConvergencePoint": "Boolean or null - For Loop Lf: Does it have convergence point at top of U?",
      "convergenceRidgeCount": "Integer or null - For Loop Lf: How many ridges converge? (should be ≥4)",
      "convergenceAngle": "String or null - For Loop Lf: 'Acute angle' if convergence exists and ridges DO NOT flow out",
      "hasEnclosedShape": "Boolean or null - For Arch Ae: Does bottom arc enclose a triangle/circle?",
      "enclosedShapeType": "String or null - For Arch Ae: 'Triangle', 'Circle', 'White block', or null",
      "notableFeatures": [
        {
          "type": "String (e.g., Bifurcation, Ridge Ending, Convergence Point, Enclosed Shape, S-Line, Inner Delta, Outer Delta)",
          "location": "String (e.g., Upper-left, Near core, Center, Top of U, Bottom arc)",
          "description": "String - Detailed description of this feature (REQUIRED)"
        }
      ]
    },
    "imageQuality": {
      "clarity": "String (High, Medium, Low, Blurry, Partial)",
      "assessment": "String"
    }
  }
}"""
    
    def analyze_fingerprint(self, image_path: Path, image_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        分析指纹图像并返回结构化JSON数据
        
        Args:
            image_path: 图像文件路径
            image_id: 图像ID（可选）
        
        Returns:
            包含分析结果的字典，如果失败则返回None
        """
        try:
            # 验证文件存在
            if not Path(image_path).exists():
                logger.error(f"图像文件不存在: {image_path}")
                return None
            
            # 编码图像
            logger.info(f"正在编码图像: {image_path}")
            base64_image = self.encode_image(image_path)
            
            # 获取分析提示词
            prompt = self.get_analysis_prompt()
            
            # 调用Gemini API
            logger.info("正在调用Gemini API进行分析...")
            response = self.client.chat.completions.create(
                model="gemini-2.5-flash",
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
                        ]
                    }
                ],
                temperature=0.1
            )
            
            # 提取响应内容
            content = response.choices[0].message.content
            logger.info("API响应已获取")

            # 清理响应内容（移除markdown代码块标记）
            if content.startswith("```json"):
                content = content[7:]  # 移除 ```json
            if content.startswith("```"):
                content = content[3:]  # 移除 ```
            if content.endswith("```"):
                content = content[:-3]  # 移除末尾的 ```
            content = content.strip()

            # 解析JSON
            analysis_result = json.loads(content)
            
            # 添加图像ID（如果提供）
            if image_id and "fingerprintAnalysis" in analysis_result:
                analysis_result["fingerprintAnalysis"]["imageId"] = image_id
            
            logger.info("指纹分析完成")
            return analysis_result
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            logger.error(f"原始响应: {content if 'content' in locals() else 'N/A'}")
            return None
        except Exception as e:
            logger.error(f"分析过程中出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def _build_subtype_distribution_from_analyzed(annotation_dir, output_dir):
        """
        根据output_dir中已分析的样本，统计其annotation中的子纹型分布

        Args:
            annotation_dir: annotation文件目录
            output_dir: 输出结果目录（已分析的样本）

        Returns:
            (subtype_counts, average_count, minority_subtypes)
        """
        from collections import defaultdict

        subtype_counts = defaultdict(int)
        annotation_dir = Path(annotation_dir)
        output_dir = Path(output_dir)

        if not annotation_dir.exists():
            logger.warning(f"Annotation目录不存在: {annotation_dir}")
            return {}, 0, set()

        if not output_dir.exists():
            logger.warning(f"Output目录不存在: {output_dir}")
            return {}, 0, set()

        # 获取output_dir中已分析的样本文件名
        analyzed_files = {f.stem for f in output_dir.glob("*.txt")}

        if not analyzed_files:
            logger.warning("Output目录中没有已分析的样本")
            return {}, 0, set()

        # 只统计已分析样本对应的annotation
        for ann_file in annotation_dir.glob("*.json"):
            try:
                # 只统计已分析的样本
                if ann_file.stem not in analyzed_files:
                    continue

                with open(ann_file, 'r', encoding='utf-8') as f:
                    ann_data = json.load(f)

                # 获取f_code（包含主纹型和子纹型）
                f_code = ann_data.get('f_code', '')
                if f_code:
                    subtype_counts[f_code] += 1
            except Exception as e:
                logger.debug(f"读取annotation失败 {ann_file}: {e}")

        if not subtype_counts:
            logger.warning("未找到任何已分析样本的annotation数据")
            return {}, 0, set()

        # 计算平均值
        average_count = sum(subtype_counts.values()) / len(subtype_counts)

        # 找出少数桶（低于平均值）
        minority_subtypes = {subtype for subtype, count in subtype_counts.items()
                            if count < average_count}

        logger.info(f"📊 已分析样本的子纹型分布统计 (样本数: {sum(subtype_counts.values())}):")
        logger.info(f"   总子纹型数: {len(subtype_counts)}")
        logger.info(f"   平均样本数: {average_count:.1f}")
        logger.info(f"   少数桶数: {len(minority_subtypes)}")
        logger.info(f"   子纹型分布: {dict(sorted(subtype_counts.items()))}")

        return dict(subtype_counts), average_count, minority_subtypes

    def analyze_batch(self, image_dir: Path, output_dir: Optional[Path] = None,
                     annotation_dir: Optional[Path] = None, balance_by_subtype: bool = False,
                     batch_size: int = 100) -> Dict[str, Any]:
        """
        批量分析指纹图像（支持断点续处理和长尾数据平衡）

        Args:
            image_dir: 包含指纹图像的目录
            output_dir: 输出结果的目录（可选）
            annotation_dir: annotation文件目录（用于统计子纹型分布）
            balance_by_subtype: 是否只分析少数桶的样本
            batch_size: 每处理多少个样本后重新统计一次桶的分布（默认100）

        Returns:
            包含所有分析结果的字典
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_images": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "skipped_analyses": 0,
            "skipped_majority": 0,
            "analyses": [],
            "batch_updates": []
        }

        image_dir = Path(image_dir)
        if not image_dir.exists():
            logger.error(f"目录不存在: {image_dir}")
            return results

        # 如果启用长尾平衡，先统计已分析样本的子纹型分布
        minority_subtypes = set()
        if balance_by_subtype and annotation_dir and output_dir:
            _, _, minority_subtypes = self._build_subtype_distribution_from_analyzed(annotation_dir, output_dir)
            logger.info(f"🎯 初始少数桶: {minority_subtypes}")

        # 查找所有图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = [f for f in image_dir.iterdir()
                      if f.suffix.lower() in image_extensions]

        results["total_images"] = len(image_files)
        logger.info(f"找到 {len(image_files)} 个图像文件")
        if balance_by_subtype:
            logger.info(f"📦 Batch大小: {batch_size}，每处理完{batch_size}个样本后重新统计桶的分布\n")

        # 分析每个图像
        processed_count = 0
        for idx, image_file in enumerate(image_files, 1):
            logger.info(f"处理 [{idx}/{len(image_files)}]: {image_file.name}")

            # 每处理完batch_size个样本，重新统计一次已分析样本的桶的分布
            if balance_by_subtype and annotation_dir and output_dir and processed_count > 0 and processed_count % batch_size == 0:
                logger.info(f"\n🔄 已处理{processed_count}个样本，重新统计已分析样本的桶的分布...")
                _, _, minority_subtypes = self._build_subtype_distribution_from_analyzed(annotation_dir, output_dir)
                logger.info(f"🎯 更新后的少数桶: {minority_subtypes}\n")
                results["batch_updates"].append({
                    "processed_count": processed_count,
                    "minority_subtypes": list(minority_subtypes)
                })

            # 检查结果文件是否已存在（断点续处理）
            if output_dir:
                output_dir_path = Path(output_dir)
                output_file = output_dir_path / f"{image_file.stem}.txt"

                if output_file.exists():
                    logger.info(f"⏭️  结果已存在，跳过: {output_file.name}")
                    results["skipped_analyses"] += 1
                    processed_count += 1
                    continue

            # 如果启用长尾平衡，检查该样本是否属于少数桶
            if balance_by_subtype and annotation_dir and minority_subtypes:
                annotation_file = Path(annotation_dir) / f"{image_file.stem}.json"
                if annotation_file.exists():
                    try:
                        with open(annotation_file, 'r', encoding='utf-8') as f:
                            ann_data = json.load(f)
                        f_code = ann_data.get('f_code', '')

                        if f_code not in minority_subtypes:
                            logger.info(f"⏭️  样本属于多数桶 ({f_code})，跳过")
                            results["skipped_majority"] += 1
                            processed_count += 1
                            continue
                    except Exception as e:
                        logger.debug(f"读取annotation失败 {annotation_file}: {e}")

            analysis = self.analyze_fingerprint(image_file, image_id=image_file.stem)

            if analysis:
                results["analyses"].append(analysis)
                results["successful_analyses"] += 1

                # 保存单个结果为txt文件（与原图像同名，后缀为.txt）
                if output_dir:
                    output_dir_path = Path(output_dir)
                    output_dir_path.mkdir(parents=True, exist_ok=True)

                    # 使用原图像的名字，但后缀改为.txt
                    output_file = output_dir_path / f"{image_file.stem}.txt"
                    try:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(analysis, f, indent=2, ensure_ascii=False)
                        logger.info(f"✅ 结果已保存到: {output_file}")
                    except Exception as e:
                        logger.error(f"❌ 保存文件失败 {output_file}: {e}")
            else:
                results["failed_analyses"] += 1

            processed_count += 1

        return results


def main():
    """示例使用"""
    import os
    import argparse

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='指纹Gemini分析器')
    parser.add_argument('--image-dir', type=str, default='images', help='图像目录')
    parser.add_argument('--output-dir', type=str, default='analysis_results', help='输出目录')
    parser.add_argument('--annotation-dir', type=str, default='annotations', help='annotation目录')
    parser.add_argument('--balance', action='store_true', help='启用长尾数据平衡（只分析少数桶）')
    parser.add_argument('--batch-size', type=int, default=100, help='每处理多少个样本后重新统计桶的分布（默认100）')
    args = parser.parse_args()

    # 获取API密钥
    api_key = os.getenv("YUNWU_API_KEY")
    if not api_key:
        logger.error("未设置YUNWU_API_KEY环境变量")
        return

    # 创建分析器
    analyzer = FingerprintGeminiAnalyzer(api_key)

    # 批量分析
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    annotation_dir = Path(args.annotation_dir) if args.balance else None

    results = analyzer.analyze_batch(
        image_dir,
        output_dir=output_dir,
        annotation_dir=annotation_dir,
        balance_by_subtype=args.balance,
        batch_size=args.batch_size
    )

    # 打印统计信息
    logger.info(f"\n📊 分析完成统计:")
    logger.info(f"   总图像数: {results['total_images']}")
    logger.info(f"   成功分析: {results['successful_analyses']}")
    logger.info(f"   失败分析: {results['failed_analyses']}")
    logger.info(f"   已存在跳过: {results['skipped_analyses']}")
    if args.balance:
        logger.info(f"   多数桶跳过: {results['skipped_majority']}")


if __name__ == "__main__":
    main()

