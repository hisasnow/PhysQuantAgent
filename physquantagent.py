import base64
import io
import os
import google.generativeai as genai
from openai import OpenAI
import json
import re
from PIL import Image
import numpy as np
import random
import time
from datetime import datetime
import subprocess
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

try:
    # Nano-banana (Gemini Image) uses the newer google-genai SDK
    from google import genai as genai2
    from google.genai import types as genai2_types
except Exception:
    genai2 = None
    genai2_types = None

load_dotenv()
# ==============================================================================
# 1. Settings
# ==============================================================================
# --- ▼▼▼ User configuration items ▼▼▼ ---
# Select API: 'gemini' or 'gpt'
API_PROVIDER = 'gpt' 

# Load API key from environment variables
# Set GEMINI_API_KEY=... or OPENAI_API_KEY=... in .env file,
# or set environment variables like export GEMINI_API_KEY="..."
if API_PROVIDER == 'gemini':
    genai.configure(api_key=os.environ.get("GEMINI_API"))
    API_MODEL_NAME = "gemini-2.5-pro" 
elif API_PROVIDER == 'gpt':
    client = OpenAI(api_key=os.environ.get("OPENAI_API"))
    API_MODEL_NAME = "gpt-5.2-2025-12-11"
else:
    raise ValueError("API_PROVIDER must be 'gemini' or 'gpt'")

JSON_DATA_PATH = "./Visphysquant/output.json"
EVAL_SAMPLING_RATIO = 5
# Wait time between API requests (seconds)
SLEEP_TIME = 1.0
MAX_WORKERS = 1 # Number of parallel executions (1 is recommended as GroundingDINO/SAM and image generation are heavy)

# --------------
# Tool settings
# --------------
TOOL_OUTPUT_ROOT = Path("outputs") / "tool_oracle" / datetime.now().strftime("%Y-%m-%d_%H%M%S")

# nano-banana (internal slice image)
NANO_BANANA_MODEL_NAME = "gemini-3-pro-image-preview"

# object detection / scale estimation (GroundingDINO+SAM+depth measure)
GDINO_SCRIPT = Path("groundingdino") / "vlm_caption_segment_measure.py"
GDINO_VLM_PROVIDER = "gemini"  # caption-to-class prompt inside the script
GDINO_VLM_MODEL = "gemini-3-flash-preview"  # used only for captioning in that script
GDINO_DEPTH_UNIT_SCALE = 1.0
GDINO_DEPTH_UNIT_NAME = "mm"

# ----------------
# Preprocessed VP images (no tool execution)
# ----------------
# Read and use pre-generated annotated images from here
# without executing tools (GroundingDINO/SAM or nano-banana generation).
VP_IMAGES_ROOT = Path("./Visual_prompt_images")
VP_BBOX_DIR = VP_IMAGES_ROOT / "bbox"
VP_AXIS_DIR = VP_IMAGES_ROOT / "axis"
VP_NANO_BANANA_DIR = VP_IMAGES_ROOT / "nano_banana"

# Same format as prepare_vpimages.py: YYYY-MM-DD--HH-MM-SS
VP_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}")

# ==============================================================================
# 2. Data Preparation
# ==============================================================================
print(f"Loading dataset from '{JSON_DATA_PATH}'...")

with open(JSON_DATA_PATH, encoding="utf-8") as f:
    full_dataset = json.load(f)

if EVAL_SAMPLING_RATIO < 1.0:
    sample_size = max(1, int(len(full_dataset) * EVAL_SAMPLING_RATIO))
    random.seed(42)
    full_dataset = random.sample(full_dataset, sample_size)

base_prompt =("You are a physics and engineering assistant.\n"
"Estimate the weight of the object from the images.\n"
)

# Tool selection prompt for multi-turn
TOOL_SELECTION_PROMPT = (
    "You are a visual reasoning assistant. "
    "You will be given an image of an object.\n"
    "Your task is to estimate the weight of the object.\n"
    "Choose which tools to use (you may choose multiple) to help estimate the object's weight.\n\n"
    "Available tools:\n"
    "- nano_banana: generate internal cross-sectional blueprint style images of the object (to infer internal structure).\n"
    "This helps to infer the internal structure (void ratio) of an object.\n"
    "Note that the generated images are not actual images of the internal structure, but are for reference purposes only.\n"
    "- object_detection: create an annotated image with detected object bounding box.\n"
    "- scale_estimation: create an annotated image with measured X/Y lengths (from depth).\n\n"
    "Use these tools only when absolutely necessary. If inference can be performed using the initial input image alone, do not employ the tools. Furthermore, the generated images should be used for reference purposes only.\n"
    "Return ONLY a JSON object with this schema:\n"
    "{\"tools\": [\"nano_banana\"|\"object_detection\"|\"scale_estimation\"], "
    "\"reason\": \"...\"}.\n"
    "Do not include any other text."
)

print(f"Total evaluation data: {len(full_dataset)}")
print(f"API Provider: {API_PROVIDER}")
print(f"Model: {API_MODEL_NAME}")
print(f"Prompt: {base_prompt}")

# ==============================================================================
# 3. Helper Functions
# ==============================================================================
def parse_weight_from_text(text: str) -> Optional[float]:
    # 0) \boxed{\text{Answer: -\,kg}} or \boxed{\text{Answer: -\,kg}}
    # Note: \\\\, in python string -> \\, in regex -> matches literal \,
    pattern0 = r'\\boxed\{\s*\\text\{[^0-9]*?(\d+(?:\.\d+)?)\s*(?:\\\\,|\s*|\\,)\s*kg\s*\}\s*\}'

    # 1) Answer: ... -kg
    pattern1 = r'(?i)answer[^0-9]*?(\d+(?:\.\d+)?)\s*kg(?![a-zA-Z])'

    # 2) Answer ... \boxed{\text{-kg}}
    pattern2 = r'(?i)answer.*?\\boxed\{\s*\\text\{\s*(\d+(?:\.\d+)?)\s*kg\s*\}\s*\}'

    # 3) Answer ... \boxed{-\text{kg}}
    pattern3 = r'(?i)answer.*?\\boxed\{\s*(\d+(?:\.\d+)?)\s*\\text\{\s*kg\s*\}\s*\}'

    # More specific -> General
    for pattern in (pattern0, pattern2, pattern3, pattern1):
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))

    return None


def image_to_base64(image: Image.Image) -> str:
    """Convert Pillow Image to Base64 encoded string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def _openai_responses_text(prompt: str, images: List[Image.Image]) -> str:
    """Generate text with OpenAI Responses API (supports image input)."""
    if not os.environ.get("OPENAI_API"):
        raise EnvironmentError("OPENAI_API is not set")

    content = [{"type": "input_text", "text": prompt}]
    for img in images:
        b64 = image_to_base64(img)
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})

    resp = client.responses.create(
        model=API_MODEL_NAME,
        input=[{"role": "user", "content": content}],
    )
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt

    # Fallback for old SDK compatibility
    out = getattr(resp, "output", None)
    if out and isinstance(out, list):
        for item in out:
            for c in item.get("content", []) or []:
                if c.get("type") == "output_text" and c.get("text"):
                    return c["text"]
    return ""


def _gemini_text(prompt: str, images: List[Image.Image]) -> str:
    api_key = os.environ.get("GEMINI_API")
    if not api_key:
        raise EnvironmentError("GEMINI_API is not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(API_MODEL_NAME)
    resp = model.generate_content([prompt] + images)
    # prompt_feedback might not exist in some SDKs, so it's best-effort
    return getattr(resp, "text", "") or ""


def vlm_generate_text(prompt: str, images: List[Image.Image]) -> str:
    if API_PROVIDER == "gpt":
        return _openai_responses_text(prompt, images)
    if API_PROVIDER == "gemini":
        return _gemini_text(prompt, images)
    raise ValueError("Unsupported API_PROVIDER")


def _safe_obj_id_from_paths(image_paths: List[str]) -> str:
    if not image_paths:
        return "unknown"
    path = str(image_paths[0])
    if "/scenes/" in path:
        match = re.search(r"/scenes/([^/]+)/", path)
        if match:
            obj_id = match.group(1)
            return re.sub(r"_scan$", "", obj_id)
    if "/record3d/" in path:
        obj_id = path.split("/record3d/")[1].split("/")[0]
        return re.sub(r"_scan$", "", obj_id)

    try:
        p = Path(path)
        if p.parent.name == "rgb" and p.parent.parent.name:
            return re.sub(r"_scan$", "", p.parent.parent.name)
        if p.parent.name:
            return re.sub(r"_scan$", "", p.parent.name)
    except Exception:
        pass

    # fallback: parent folder
    parts = [part for part in path.split("/") if part]
    if len(parts) >= 3 and parts[-2] == "rgb":
        return re.sub(r"_scan$", "", parts[-3])
    return re.sub(r"_scan$", "", parts[-2]) if len(parts) >= 2 else "unknown"


def _find_preprocessed_tool_image(tool_name: str, image_paths: List[str]) -> Optional[Path]:
    """Resolve tool output from Visual_prompt_images.

    tool_name:
      - nano_banana      -> Visual_prompt_images/internal_image/<obj_id>_sliced.png
      - object_detection -> Visual_prompt_images/bbox/<obj_id>.bbox*.jpg
      - scale_estimation -> Visual_prompt_images/axis/<obj_id>.axis*.jpg
    """
    obj_id = _safe_obj_id_from_paths(image_paths)
    if not obj_id or obj_id == "unknown":
        return None

    if tool_name == "object_detection":
        if not VP_BBOX_DIR.exists():
            return None
        cands = sorted(VP_BBOX_DIR.glob(f"{obj_id}.bbox*.jpg"))
        return cands[0] if cands else None

    if tool_name == "scale_estimation":
        if not VP_AXIS_DIR.exists():
            return None
        cands = sorted(VP_AXIS_DIR.glob(f"{obj_id}.axis*.jpg"))
        return cands[0] if cands else None

    if tool_name == "nano_banana":
        if not VP_NANO_BANANA_DIR.exists():
            return None
        # Usually <obj_id>_sliced.png. Search with prefix just in case.
        p = VP_NANO_BANANA_DIR / f"{obj_id}_sliced.png"
        if p.exists():
            return p
        cands = sorted(VP_NANO_BANANA_DIR.glob(f"{obj_id}*_sliced.png"))
        return cands[0] if cands else None

    return None


def _load_images(image_paths: List[str], *, limit: int = None, resize_half: bool = True) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    # If limit is None or < 0, target all images (no slice range)
    if limit is None or limit < 0:
        target_paths = image_paths
    else:
        target_paths = image_paths[:limit]

    for p in target_paths:
        try:
            img = Image.open(p).convert("RGB")
            if resize_half:
                img = img.resize((max(1, img.width // 2), max(1, img.height // 2)))
            imgs.append(img)
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return imgs


def select_tools_vlm(image_list: List[Image.Image]) -> Tuple[List[str], str, str]:
    """1st turn: Show images and let it select tools.

    Returns: (tools, reason, raw_json_text)
    """
    raw = vlm_generate_text(TOOL_SELECTION_PROMPT, image_list)
    raw = raw.strip()

    # Remove Markdown code block syntax if present
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except Exception:
        # Fallback if extra text is mixed in
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                # Use default settings if parsing fails
                data = {"tools": ["scale_estimation"], "reason": "JSON parse failed"}
        else:
            return ["scale_estimation"], "fallback: could not parse JSON", raw

    tools = data.get("tools", [])
    if not isinstance(tools, list):
        tools = []
    tools = [str(t).strip() for t in tools]
    allow = {"nano_banana", "object_detection", "scale_estimation"}
    tools = [t for t in tools if t in allow]
    # if not tools:
    #     tools = ["scale_estimation"]
    reason = str(data.get("reason", "")).strip()
    return tools, reason, raw

# ==============================================================================
# 4. Execute Inference
# ==============================================================================
def process_example(example):
    """
    Process a single sample and return the result
    """
    try:
        image_paths = example.get("images", [])
        obj_id = _safe_obj_id_from_paths(image_paths)

        if len(image_paths) < 1:
            return {"error_message": "Skipped because there are no images"}

        # --- Turn 1: tool selection ---
        # Get 0th, 3rd, 7th images (check if indices exist)
        indices_to_load = [0, 3, 7]
        selected_paths = []
        for idx in indices_to_load:
            if idx < len(image_paths):
                selected_paths.append(image_paths[idx])
        
        first_img_list = _load_images(selected_paths, limit=None, resize_half=True)

        if not first_img_list:
            return {"error_message": "Skipped because the first image cannot be read"}
        selected_tools, tool_reason, tool_choice_raw = select_tools_vlm(first_img_list)

        # --- Run tools ---
        tool_outputs: Dict[str, str] = {}
        tool_errors: Dict[str, str] = {}

        # NOTE:
        # We do not "execute" the tool here, but load the pre-generated images from Visual_prompt_images.
        # (Avoids heavy processing of GroundingDINO/SAM or nano-banana)

        if "nano_banana" in selected_tools:
            p = _find_preprocessed_tool_image("nano_banana", image_paths)
            if p is not None and p.exists():
                tool_outputs["nano_banana"] = str(p)
            else:
                tool_errors["nano_banana"] = f"preprocessed image not found under {VP_NANO_BANANA_DIR} (obj_id={obj_id})"

        if "object_detection" in selected_tools:
            p = _find_preprocessed_tool_image("object_detection", image_paths)
            if p is not None and p.exists():
                tool_outputs["object_detection"] = str(p)
            else:
                tool_errors["object_detection"] = f"preprocessed image not found under {VP_BBOX_DIR} (obj_id={obj_id})"

        if "scale_estimation" in selected_tools:
            p = _find_preprocessed_tool_image("scale_estimation", image_paths)
            if p is not None and p.exists():
                tool_outputs["scale_estimation"] = str(p)
            else:
                tool_errors["scale_estimation"] = f"preprocessed image not found under {VP_AXIS_DIR} (obj_id={obj_id})"

        # tool selection / tool run logs are printed in the summary section (after inference)

        # --- Turn 2: weight inference ---
        # Base multi-view images
        # All images without limit
        images = _load_images(image_paths, limit=None, resize_half=True)

        if not images:
            return {"error_message": "Skipped because there are no valid images"}

        head_imgs: List[Image.Image] = []
        tail_imgs: List[Image.Image] = []

        # object_detection is input first
        if "object_detection" in tool_outputs:
            p = tool_outputs["object_detection"]
            if p and os.path.exists(p):
                try:
                    head_imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    pass

        for k in ["nano_banana", "scale_estimation"]:
            p = tool_outputs.get(k)
            if p and os.path.exists(p):
                try:
                    tail_imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    pass

        predicted_process = ""
        max_retries = 3
        grid_unit = example.get("grid_unit", "N/A")
        caption = example.get("caption", "N/A")
        current_prompt = base_prompt.format(grid_unit=grid_unit, caption=caption)

        # Append tool selection results and descriptions of tool outputs (text)
        tool_text = ("")

        if "scale_estimation" in selected_tools:
             tool_text += "\nThere is an image with measured dimensions (axis visualization) for scale reference.This shows the vertical and horizontal size of the bbox that surrounds the object."

        if "nano_banana" in selected_tools:
             tool_text +=("The last image is a sliced image generated for reference to infer the internal structure(void ration). \n"
                          "Use this cross-sectional image to estimate the porosity and calculate the volume of the object.\n"
                         "Note that the generated images are not actual images of the internal structure, but are for reference purposes only. \n")

        #current_prompt = current_prompt + tool_text
        current_prompt = current_prompt + tool_text + \
            "The reference image may have measurement errors or hallucinations\n." + \
            "If the estimated value deviates from the generally accepted value, please estimate the general value.\n" + \
            "Reason step by step and finally state your answer in kilograms like 'Answer:-kg'.\n"


        for attempt in range(max_retries):
            try:
                predicted_process = vlm_generate_text(current_prompt, head_imgs + images + tail_imgs)
                if predicted_process:
                    break
            except Exception as e:
                print(f"Error occurred during API call (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt + 1 == max_retries:
                    return {"error_message": f"API call failed {max_retries} times: {e}", "failed_example": example}
                time.sleep(2 ** attempt)

        predicted_weight_parsed = parse_weight_from_text(predicted_process)
        parsing_failed = predicted_weight_parsed is None
        predicted_weight = predicted_weight_parsed if not parsing_failed else 0.0
        true_weight = example.get("weight_kg", None)

        error = ade = alde = mnre = None
        if true_weight is not None and true_weight > 0:
            error = np.abs(true_weight - predicted_weight) / true_weight
            ade = np.abs(true_weight - predicted_weight)
            if predicted_weight > 0:
                alde = np.abs(np.log(true_weight) - np.log(predicted_weight))
                mnre = min(true_weight / predicted_weight, predicted_weight / true_weight)

        time.sleep(SLEEP_TIME)

        return {
            "obj_id": obj_id,
            "true_weight": true_weight, "predicted_weight": predicted_weight,
            "predicted_process": predicted_process,
            "selected_tools": selected_tools,
            "tool_reason": tool_reason,
            "tool_outputs": tool_outputs,
            "tool_errors": tool_errors,
            "error": error, "ade": ade, "alde": alde, "mnre": mnre,
            "parsing_failed": parsing_failed,
            "image_paths": image_paths
        }

    except Exception as e:
        return {"error_message": str(e)}

results = []
failed_samples = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit tasks
    future_to_example = {executor.submit(process_example, example): example for example in full_dataset}
    
    # Visualize progress with tqdm
    for future in tqdm(as_completed(future_to_example), total=len(full_dataset), desc="Processing Inference"):
        result = future.result()
        if "error_message" in result:
            print(f"Error occurred: {result['error_message']}")
            if "failed_example" in result:
                failed_samples.append(result['failed_example'])
        else:
            results.append(result)

# ==============================================================================
# 5. Display Results and Summary
# ==============================================================================
errors, ade_list, alde_list, mnre_list = [], [], [], []

print("\n" + "="*60)
print("                  Individual Inference Results                  ")
print("="*60)
for i, res in enumerate(results):
    print(f"\n--- Sample {i+1}/{len(results)} ---")
    
    extracted_id = _safe_obj_id_from_paths(res.get('image_paths') or [])
    
    print(f"Object ID: {extracted_id}")
    print(f"True Weight: {res['true_weight']:.4f} kg" if res['true_weight'] is not None else "True Weight: N/A")
    print(f"Predicted Weight: {res['predicted_weight']:.4f} kg" if res['predicted_weight'] is not None else "Predicted Weight: Not found")
    if res['error'] is not None: print(f"Absolute Percentage Error: {res['error']:.2%}")
    if res['mnre'] is not None: print(f"MnRE: {res['mnre']:.2%}")
    print("--- Model Output ---")

    # Tool selection/execution logs (displayed right before model output)
    if 'selected_tools' in res:
        print("[ToolSelection]")
        print(f"obj_id={res.get('obj_id', 'N/A')}")
        print(f"selected_tools={res.get('selected_tools')}")
        if res.get('tool_reason'):
            print(f"reason={res.get('tool_reason')}")
        if res.get('tool_outputs'):
            print("tool_outputs=")
            for k, v in (res.get('tool_outputs') or {}).items():
                print(f"  - {k}: {v}")
        if res.get('tool_errors'):
            print("tool_errors=")
            for k, v in (res.get('tool_errors') or {}).items():
                print(f"  - {k}: {v}")
    print(res['predicted_process'])
    
    if res['error'] is not None: errors.append(res['error'])
    if res['ade'] is not None: ade_list.append(res['ade'])
    if res['alde'] is not None: alde_list.append(res['alde'])
    if res['mnre'] is not None: mnre_list.append(res['mnre'])

# ==============================================================================
# 6. Summary
# ==============================================================================
print("\n" + "="*60)
print("                  Inference Evaluation Summary                  ")
print("="*60)

if failed_samples:
    print("\n" + "="*60)
    print("                  API Call Failed Samples                  ")
    print("="*60)
    print(f"Failed to process {len(failed_samples)} samples.")
    # Output details of failed samples (e.g. ID, image path)
    for i, sample in enumerate(failed_samples):
        print(f"--- Failed Sample {i+1} ---")
        # Display if there's a key to identify the sample (e.g. 'item_id', 'images')
        if "item_id" in sample:
            print(f"  ID: {sample['item_id']}")
        if "images" in sample:
            print(f"  Image Path: {sample['images'][:1]}...") # Display only the first image path
    print("="*60)

parsing_failed_indices = [
    i + 1 for i, res in enumerate(results) if res.get('parsing_failed', False)
]
if parsing_failed_indices:
    print("\n" + "-"*35)
    print("      Extraction Failed Sample Numbers      ")
    print("-" * 35)
    print(", ".join(map(str, parsing_failed_indices)))
    print("-" * 35)

successful_predictions = sum(1 for res in results if not res.get('parsing_failed', False))
total_samples = len(full_dataset)
success_rate = (successful_predictions / total_samples) * 100 if total_samples > 0 else 0

mean_mape = np.mean(errors) if errors else 0
mean_ade = np.mean(ade_list) if ade_list else 0
mean_alde = np.mean(alde_list) if alde_list else 0
mean_mnre = np.mean(mnre_list) if mnre_list else 0

print(f"Total Evaluation Samples: {total_samples}")
print(f"Successful Prediction Parses: {successful_predictions} ({success_rate:.1f}%)")

if successful_predictions > 0:
    errors_with_indices = [
        (res['error'], i)
        for i, res in enumerate(results)
        if res['error'] is not None
    ]
    errors_with_indices.sort(key=lambda x: x[0], reverse=True)

    print("\n" + "-"*35)
    print("    Top 5 Worst Absolute Percentage Errors    ")
    print("-" * 35)
    for error, index in errors_with_indices[:5]:
        obj_id = results[index].get("obj_id", "N/A")
        print(f"Object ID: {obj_id}, Error: {error:.2%}")
    print("-" * 35)

if successful_predictions > 0:
    print("-" * 35)
    print(f"Mean Absolute Percentage Error (MAPE): {mean_mape:.2%}")
    print(f"Average Absolute Error (ADE):           {mean_ade:.4f} kg")
    print(f"Average Absolute Log Error (ALDE):    {mean_alde:.4f}")
    print(f"Mean Normalized Ratio Error (MNRE):      {mean_mnre:.4f}")
    print("-" * 35)
    
print(f"API Provider: {API_PROVIDER}")
print(f"Model: {API_MODEL_NAME}")
print(f"Prompt: {base_prompt}")
print(f"Input File: {JSON_DATA_PATH }")

print("="*60)
