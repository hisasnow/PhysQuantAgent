import argparse
import base64
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from dotenv import load_dotenv


def _add_groundingdino_repo_to_syspath(groundingdino_repo: Path) -> None:
    groundingdino_repo = groundingdino_repo.resolve()
    if not groundingdino_repo.exists():
        raise FileNotFoundError(
            f"GroundingDINO repo not found: {groundingdino_repo}. "
            "Clone it or pass --groundingdino-repo."
        )
    sys.path.insert(0, str(groundingdino_repo))


def _safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name).strip("_")


def _encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def _ensure_rgb_path(image_path: str) -> str:
    if image_path.startswith("./"):
        rel = image_path.replace("./", "")
        if os.path.exists(rel):
            return rel
    return image_path


def _infer_record3d_base_dir(rgb_path: Path) -> Path:
    return rgb_path.parent.parent


def _load_intrinsics_K(base_dir: Path) -> List[float]:
    intrinsics_path = base_dir / "camera_intrinsics.json"
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Intrinsics not found: {intrinsics_path}")
    intr = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    K = intr.get("K")
    if not isinstance(K, list) or len(K) != 9:
        raise ValueError(f"Invalid intrinsics K: {K}")
    return [float(x) for x in K]


def _load_depth_map(base_dir: Path, rgb_path: Path) -> np.ndarray:
    depth_path = base_dir / "depth" / f"{rgb_path.stem}.png"
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth map not found: {depth_path}")
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth: {depth_path}")
    return depth


def _pixel_to_3d(u: float, v: float, z: float, K: List[float]) -> np.ndarray:
    # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    fx = K[0]
    cx = K[2]
    fy = K[4]
    cy = K[5]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.asarray([x, y, z], dtype=np.float64)


def _depth_at_rgb_px(
    depth_img: np.ndarray,
    *,
    rgb_u: float,
    rgb_v: float,
    scale_factor: float,
    patch: int = 2,
) -> Optional[float]:
    """Return depth at (u,v) in RGB coordinates.

    Uses median of non-zero depth in a (2*patch+1)^2 neighborhood on the depth image.
    """

    if scale_factor <= 0:
        raise ValueError("scale_factor must be > 0")

    # Reject clearly invalid RGB coordinates early.
    # (In normal usage endpoints are within the RGB image; this is just a guard.)
    if rgb_u < 0 or rgb_v < 0:
        return None

    h, w = depth_img.shape[:2]

    # Map RGB pixel coordinates to depth pixel coordinates.
    # NOTE: Using round() alone can push border pixels out of range (e.g. rgb_u=rgb_w-1),
    # so we clip to [0, w-1]/[0, h-1] to avoid spurious None at endpoints.
    d_u = int(round(rgb_u / scale_factor))
    d_v = int(round(rgb_v / scale_factor))
    d_u = int(np.clip(d_u, 0, w - 1))
    d_v = int(np.clip(d_v, 0, h - 1))

    u0 = max(0, d_u - patch)
    u1 = min(w - 1, d_u + patch)
    v0 = max(0, d_v - patch)
    v1 = min(h - 1, d_v + patch)
    patch_vals = depth_img[v0 : v1 + 1, u0 : u1 + 1].reshape(-1)
    patch_vals = patch_vals[patch_vals > 0]
    if patch_vals.size == 0:
        return None
    return float(np.median(patch_vals))


def _enhance_class_name(class_names: List[str]) -> List[str]:
    return [f"all {class_name}s" for class_name in class_names]


def _caption_to_classes_vlm(
    *,
    image_path: Path,
    provider: str,
    model_name: str,
    max_classes: int,
) -> Tuple[str, List[str]]:
    """Generate caption/classes using a VLM.

    Returns: (raw_text, classes)
    """

    # IMPORTANT: downstream segmentation uses only ONE word prompt.
    prompt = (
        "You are given an image. Identify the single most prominent physical object. "
        "Return ONLY a JSON object with:\n"
        '  - "word": a single lowercase English noun (ONE WORD ONLY, no spaces, no punctuation)\n'
        "Do not include any other keys."
    )

    content = ""
    if provider == "gpt":
        from openai import OpenAI

        client = OpenAI(api_key=os.environ.get("OPENAI_API"))
        if not os.environ.get("OPENAI_API"):
            raise EnvironmentError("OPENAI_API is not set")
        b64 = _encode_image_base64(image_path)

        # NOTE:
        # Some models (including some gpt-5.*) do NOT accept `image_url` via the Chat Completions API.
        # In that case the server returns:
        #   "Invalid content type. image_url is only supported by certain models."
        # We first try the newer Responses API (preferred for multimodal), then fall back.
        image_data_url = f"data:image/jpeg;base64,{b64}"

        # 1) Try Responses API (multimodal)
        try:
            resp = client.responses.create(
                model=model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": image_data_url},
                        ],
                    }
                ],
                # best-effort JSON enforcement (SDK/version dependent)
                text={"format": {"type": "json_object"}},
            )
            # `output_text` is available in recent SDKs
            content = getattr(resp, "output_text", None) or ""
            if not content:
                # fallback extraction
                out = getattr(resp, "output", None)
                if out and out[0].get("content"):
                    content = out[0]["content"][0].get("text", "")
        except Exception:
            content = ""

        # 2) Fallback to Chat Completions (may fail for some models)
        if not content:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    }
                ],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
    elif provider == "gemini":
        import google.generativeai as genai
        from PIL import Image

        api_key = os.environ.get("GEMINI_API") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API (or GOOGLE_API_KEY) is not set")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        img = Image.open(image_path)
        generation_config = {"response_mime_type": "application/json"}
        resp = model.generate_content([prompt, img], generation_config=generation_config)
        content = resp.text
    else:
        raise ValueError("provider must be 'gpt' or 'gemini'")

    data = json.loads(content)
    word = str(data.get("word", "")).strip().lower()
    # enforce one token-ish (best effort)
    word = word.split()[0] if word else "object"
    return content, [word]


def _pick_main_mask(detections, strategy: str = "best_score") -> Optional[np.ndarray]:
    if getattr(detections, "mask", None) is None or len(detections.mask) == 0:
        return None
    if strategy == "union":
        out = np.zeros_like(detections.mask[0], dtype=bool)
        for m in detections.mask:
            out |= m.astype(bool)
        return out
    if strategy == "best_score":
        conf = getattr(detections, "confidence", None)
        if conf is None or len(conf) != len(detections.mask):
            return detections.mask[0].astype(bool)
        idx = int(np.argmax(np.asarray(conf)))
        return detections.mask[idx].astype(bool)
    raise ValueError("strategy must be 'best_score' or 'union'")


def _select_bbox_index(detections, strategy: str = "best_score") -> Optional[int]:
    xyxy = getattr(detections, "xyxy", None)
    if xyxy is None or len(xyxy) == 0:
        return None
    if strategy == "smallest":
        areas = []
        for box in xyxy:
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1))
        return int(np.argmin(np.asarray(areas)))
    if strategy == "best_score":
        conf = getattr(detections, "confidence", None)
        if conf is None or len(conf) != len(xyxy):
            return 0
        return int(np.argmax(np.asarray(conf)))
    raise ValueError("strategy must be 'best_score' or 'smallest'")


def _filter_detections_by_index(detections, idx: int) -> None:
    # Keep only the selected detection in-place
    detections.xyxy = detections.xyxy[[idx]]
    if getattr(detections, "confidence", None) is not None:
        detections.confidence = detections.confidence[[idx]]
    if getattr(detections, "class_id", None) is not None:
        detections.class_id = detections.class_id[[idx]]
    if getattr(detections, "mask", None) is not None:
        detections.mask = detections.mask[[idx]]


def _longest_run_endpoints(mask: np.ndarray, axis: str) -> Optional[Tuple[int, int, int, int]]:
    """Return endpoints (x1,y1,x2,y2) of the longest chord aligned with axis inside mask.

    axis='x': scan rows, find row with max (max_x-min_x)
    axis='y': scan cols, find col with max (max_y-min_y)
    """

    if mask.dtype != bool:
        mask = mask.astype(bool)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None

    h, w = mask.shape
    if axis == "x":
        best = None
        best_len = -1
        for y in range(h):
            row = mask[y]
            idx = np.where(row)[0]
            if idx.size == 0:
                continue
            x1 = int(idx.min())
            x2 = int(idx.max())
            length = x2 - x1
            if length > best_len:
                best_len = length
                best = (x1, y, x2, y)
        return best
    if axis == "y":
        best = None
        best_len = -1
        for x in range(w):
            col = mask[:, x]
            idx = np.where(col)[0]
            if idx.size == 0:
                continue
            y1 = int(idx.min())
            y2 = int(idx.max())
            length = y2 - y1
            if length > best_len:
                best_len = length
                best = (x, y1, x, y2)
        return best
    raise ValueError("axis must be 'x' or 'y'")


def _draw_line(img_bgr: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    cv2.arrowedLine(img_bgr, p1, p2, color, thickness=8, tipLength=0.05)
    cv2.circle(img_bgr, p1, 8, color, -1)


def _draw_dashed_line(
    img_bgr: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 10,
    gap_len: int = 6,
) -> None:
    """Draw a dashed line (approx) between p1 and p2."""

    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist = float(np.hypot(dx, dy))
    if dist <= 0:
        return
    ux = dx / dist
    uy = dy / dist
    step = dash_len + gap_len
    n = int(dist // step) + 1
    for i in range(n):
        s = i * step
        e = min(dist, s + dash_len)
        sx = int(round(x1 + ux * s))
        sy = int(round(y1 + uy * s))
        ex = int(round(x1 + ux * e))
        ey = int(round(y1 + uy * e))
        cv2.line(img_bgr, (sx, sy), (ex, ey), color, thickness)


def _mask_extrema_points(mask: np.ndarray) -> Optional[Dict[str, Any]]:
    """Compute 4 extremal points inside the mask and the bbox extrema.

    Returns:
      - min_x, max_x, min_y, max_y
      - min_x_point, max_x_point, min_y_point, max_y_point (guaranteed inside mask)
      - origin (min_x, max_y)  (bbox corner; may be outside mask depending on shape)
      - x_axis_end (max_x, max_y)
      - y_axis_end (min_x, min_y)
    """

    if mask.dtype != bool:
        mask = mask.astype(bool)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None

    min_x = int(xs.min())
    max_x = int(xs.max())
    min_y = int(ys.min())
    max_y = int(ys.max())

    ys_at_min_x = ys[xs == min_x]
    ys_at_max_x = ys[xs == max_x]
    xs_at_min_y = xs[ys == min_y]
    xs_at_max_y = xs[ys == max_y]

    # pick representative points on each extremal line (inside mask)
    min_x_point = (min_x, int(np.median(ys_at_min_x)))
    max_x_point = (max_x, int(np.median(ys_at_max_x)))
    min_y_point = (int(np.median(xs_at_min_y)), min_y)
    max_y_point = (int(np.median(xs_at_max_y)), max_y)

    origin = (min_x, max_y)
    x_axis_end = (max_x, max_y)
    y_axis_end = (min_x, min_y)

    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "min_x_point": min_x_point,
        "max_x_point": max_x_point,
        "min_y_point": min_y_point,
        "max_y_point": max_y_point,
        "origin": origin,
        "x_axis_end": x_axis_end,
        "y_axis_end": y_axis_end,
    }


def _estimate_depth_for_pixel_to_metric(
    *,
    depth_img: np.ndarray,
    mask: np.ndarray,
    scale_factor: float,
    origin: Tuple[int, int],
    max_samples: int = 2000,
) -> Optional[float]:
    """Pick a robust depth Z to convert pixel spans into metric spans.

    Preference:
      1) depth at origin (even if origin is outside mask, try anyway)
      2) median depth over sampled mask pixels
    """

    z0 = _depth_at_rgb_px(depth_img, rgb_u=float(origin[0]), rgb_v=float(origin[1]), scale_factor=scale_factor)
    if z0 is not None:
        return float(z0)

    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0:
        return None

    n = xs.size
    if n > max_samples:
        idx = np.random.choice(n, size=max_samples, replace=False)
        xs = xs[idx]
        ys = ys[idx]

    zs: List[float] = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        z = _depth_at_rgb_px(depth_img, rgb_u=float(x), rgb_v=float(y), scale_factor=scale_factor)
        if z is not None:
            zs.append(float(z))
    if not zs:
        return None
    return float(np.median(np.asarray(zs)))


def _pixel_span_to_metric(
    *,
    delta_px: float,
    axis: str,
    z_raw: float,
    K: List[float],
    scale_factor: float,
    depth_unit_scale: float,
) -> float:
    """Convert 1D pixel span (in RGB pixels) into metric using pinhole model at depth Z.

    We use depth-frame intrinsics, so we first convert pixel span to depth-frame span:
      delta_depth_px = delta_rgb_px / scale_factor

    Then:
      metric_x = delta_depth_px * Z / fx
      metric_y = delta_depth_px * Z / fy

    Z is converted by depth_unit_scale (e.g. mm->m is 0.001).
    """

    z = float(z_raw) * float(depth_unit_scale)
    delta_depth = float(delta_px) / float(scale_factor)
    fx = float(K[0])
    fy = float(K[4])
    if axis == "x":
        return abs(delta_depth) * z / fx
    if axis == "y":
        return abs(delta_depth) * z / fy
    raise ValueError("axis must be 'x' or 'y'")


def _metric_distance_between_rgb_pixels(
    *,
    depth_img: np.ndarray,
    p1_rgb: Tuple[int, int],
    p2_rgb: Tuple[int, int],
    scale_factor: float,
    depth_unit_scale: float,
    K: List[float],
) -> Optional[float]:
    """Metric distance between two RGB pixel coordinates using per-endpoint depth.

    Notes:
      - Depth is sampled in depth image space via `scale_factor`.
      - Intrinsics K are assumed to be for the depth frame, so (u,v) are scaled to depth pixels.
      - Depth is converted by `depth_unit_scale` (e.g. mm->m is 0.001).
    """

    x1, y1 = int(p1_rgb[0]), int(p1_rgb[1])
    x2, y2 = int(p2_rgb[0]), int(p2_rgb[1])

    z1 = _depth_at_rgb_px(depth_img, rgb_u=float(x1), rgb_v=float(y1), scale_factor=scale_factor)
    z2 = _depth_at_rgb_px(depth_img, rgb_u=float(x2), rgb_v=float(y2), scale_factor=scale_factor)
    if z1 is None or z2 is None:
        return None

    du1, dv1 = float(x1) / float(scale_factor), float(y1) / float(scale_factor)
    du2, dv2 = float(x2) / float(scale_factor), float(y2) / float(scale_factor)

    p1 = _pixel_to_3d(du1, dv1, float(z1) * float(depth_unit_scale), K)
    p2 = _pixel_to_3d(du2, dv2, float(z2) * float(depth_unit_scale), K)
    return float(np.linalg.norm(p2 - p1))


def _normalize_x_segment(ep: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = ep
    if x2 < x1:
        x1, x2 = x2, x1
    return (x1, y1, x2, y2)


def _normalize_y_segment(ep: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = ep
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def _translate_segments_to_origin(
    *,
    x_ep: Tuple[int, int, int, int],
    y_ep: Tuple[int, int, int, int],
) -> Dict[str, Any]:
    """Translate the 2 segments so they share the requested origin.

    User requirement:
      - origin.x := min x-coordinate of the X-axis segment
      - origin.y := max y-coordinate of the Y-axis segment
      - translate X segment vertically to origin.y
      - translate Y segment horizontally to origin.x

    Returns dict with shifted endpoints and origin.
    """

    x_ep = _normalize_x_segment(x_ep)
    y_ep = _normalize_y_segment(y_ep)

    x1, xy, x2, _ = x_ep
    yx, y1, _, y2 = y_ep  # y2 is max y

    origin = (x1, y2)

    # Shift X segment vertically to y=origin.y
    dx_x, dy_x = 0, origin[1] - xy
    x_shifted = (x1, xy + dy_x, x2, xy + dy_x)

    # Shift Y segment horizontally to x=origin.x
    dx_y, dy_y = origin[0] - yx, 0
    y_shifted = (yx + dx_y, y1, yx + dx_y, y2)

    return {
        "origin": origin,
        "x_original": x_ep,
        "y_original": y_ep,
        "x_shifted": x_shifted,
        "y_shifted": y_shifted,
        "shift_x": (dx_x, dy_x),
        "shift_y": (dx_y, dy_y),
    }


def _put_length_label(
    img_bgr: np.ndarray,
    *,
    text: str,
    anchor_xy: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    x, y = anchor_xy
    x = max(0, min(img_bgr.shape[1] - 1, x))
    y = max(0, min(img_bgr.shape[0] - 1, y))
    # Outline for readability: draw thicker black text first as background
    cv2.putText(
        img_bgr,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        img_bgr,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
        cv2.LINE_AA,
    )


def _shrink_mask(mask: np.ndarray, shrink_px: int) -> np.ndarray:
    """Erode mask by ~shrink_px pixels.

    Uses a single erosion with a (2*shrink_px+1) kernel, which is a good approximation of
    shrinking the boundary inward by shrink_px.
    """

    if shrink_px <= 0:
        return mask.astype(bool)

    mask_u8 = (mask.astype(np.uint8) * 255)
    k = 2 * int(shrink_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return (eroded > 0)


def _dino_sam_segment(
    *,
    rgb_bgr: np.ndarray,
    classes: List[str],
    dino_config: Path,
    dino_checkpoint: Path,
    sam_checkpoint: Path,
    sam_encoder: str,
    device: str,
    box_threshold: float,
    text_threshold: float,
    groundingdino_repo: Path,
):
    """Return detections with `.mask` computed (GroundingDINO boxes + SAM masks)."""

    _add_groundingdino_repo_to_syspath(groundingdino_repo)
    from groundingdino.util.inference import Model as GroundingDINOModel
    from segment_anything import SamPredictor, sam_model_registry

    torch_device = torch.device(device)
    dino = GroundingDINOModel(
        model_config_path=str(dino_config),
        model_checkpoint_path=str(dino_checkpoint),
        device=str(torch_device),
    )
    sam = sam_model_registry[sam_encoder](checkpoint=str(sam_checkpoint)).to(device=torch_device)
    sam_predictor = SamPredictor(sam)

    detections = dino.predict_with_classes(
        image=rgb_bgr,
        classes=_enhance_class_name(classes),
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    if len(detections) > 0 and detections.xyxy is not None and len(detections.xyxy) > 0:
        sam_predictor.set_image(rgb_rgb)
        masks: List[np.ndarray] = []
        for box in detections.xyxy:
            m, scores, _ = sam_predictor.predict(box=box.astype(np.float32), multimask_output=True)
            masks.append(m[int(np.argmax(scores))])
        detections.mask = np.asarray(masks)
    return detections


@dataclass
class MeasureResult:
    image: str
    caption_json: str
    classes: List[str]
    mask_strategy: str
    mask_shrink_px: int
    x_endpoints_rgb: Optional[List[int]]
    y_endpoints_rgb: Optional[List[int]]
    x_length: Optional[float]
    y_length: Optional[float]
    depth_unit: str
    depths: Optional[Dict[str, Optional[float]]] = None


def run_one(
    *,
    rgb_path: Path,
    output_dir: Path,
    groundingdino_repo: Path,
    dino_config: Path,
    dino_checkpoint: Path,
    sam_checkpoint: Path,
    sam_encoder: str,
    device: str,
    box_threshold: float,
    text_threshold: float,
    mask_strategy: str,
    vlm_provider: str,
    vlm_model: str,
    max_classes: int,
    depth_unit_scale: float,
    depth_unit_name: str,
    mask_shrink_px: int,
    mask_shrink_ratio: float,
    fixed_class: Optional[str] = None,
    box_select: str = "best_score",
) -> MeasureResult:
    # NOTE: The `groundingdino` Python package is imported from the cloned
    # `GroundingDINO/` repository by prepending that path to `sys.path`
    # inside `_dino_sam_segment()`.

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load RGB
    rgb_bgr = cv2.imread(str(rgb_path))
    if rgb_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {rgb_path}")
    rgb_h, rgb_w = rgb_bgr.shape[:2]

    # Load depth + intrinsics (Record3D layout)
    base_dir = _infer_record3d_base_dir(rgb_path)
    output_obj_id = base_dir.name[:-5] if base_dir.name.endswith("_scan") else base_dir.name
    visual_prompt_root = Path(__file__).resolve().parent / "Visual_prompt_images"
    bbox_export_dir = visual_prompt_root / "bbox"
    axis_export_dir = visual_prompt_root / "axis"
    bbox_export_dir.mkdir(parents=True, exist_ok=True)
    axis_export_dir.mkdir(parents=True, exist_ok=True)

    K = _load_intrinsics_K(base_dir)
    depth_img = _load_depth_map(base_dir, rgb_path)
    depth_h, depth_w = depth_img.shape[:2]
    scale_factor = rgb_w / float(depth_w)

    # VLM caption -> classes
    if fixed_class:
        caption_json = json.dumps({"word": fixed_class})
        classes = [fixed_class]
    else:
        caption_json, classes = _caption_to_classes_vlm(
            image_path=rgb_path,
            provider=vlm_provider,
            model_name=vlm_model,
            max_classes=max_classes,
        )

    detections = _dino_sam_segment(
        rgb_bgr=rgb_bgr,
        classes=classes,
        dino_config=dino_config,
        dino_checkpoint=dino_checkpoint,
        sam_checkpoint=sam_checkpoint,
        sam_encoder=sam_encoder,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        groundingdino_repo=groundingdino_repo,
    )

    # Optionally select a single bbox before any downstream processing
    sel_idx = _select_bbox_index(detections, strategy=box_select)
    if sel_idx is not None:
        _filter_detections_by_index(detections, sel_idx)

    # Save bbox-only visualization (requested)
    bbox_img = rgb_bgr.copy()
    if detections.xyxy is not None and len(detections.xyxy) > 0:
        conf = getattr(detections, "confidence", None)
        cls_ids = getattr(detections, "class_id", None)
        for i, box in enumerate(detections.xyxy):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            cv2.rectangle(bbox_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            score = float(conf[i]) if conf is not None and len(conf) > i else float("nan")
            cid = int(cls_ids[i]) if cls_ids is not None and len(cls_ids) > i else 0
            cname = classes[cid] if 0 <= cid < len(classes) else "object"
            cv2.putText(
                bbox_img,
                f"{cname} {score:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
    cv2.imwrite(str(output_dir / "annotated_bbox.jpg"), bbox_img)
    cv2.imwrite(str(bbox_export_dir / f"{output_obj_id}.bbox.jpg"), bbox_img)

    main_mask = _pick_main_mask(detections, strategy=mask_strategy)

    # Optional: shrink mask to avoid boundary depth noise
    used_shrink_px = 0
    if main_mask is not None:
        min_dim = int(min(main_mask.shape[0], main_mask.shape[1]))
        px_from_ratio = 0
        if mask_shrink_ratio and mask_shrink_ratio > 0:
            px_from_ratio = int(round(float(mask_shrink_ratio) * float(min_dim)))
        used_shrink_px = max(int(mask_shrink_px), int(px_from_ratio))
        if used_shrink_px > 0:
            main_mask = _shrink_mask(main_mask, used_shrink_px)

    # Compute extrema points (inside mask) and define axis-aligned segments
    extrema = _mask_extrema_points(main_mask) if main_mask is not None else None
    if extrema is None:
        x_ep = None
        y_ep = None
    else:
        origin = extrema["origin"]  # (min_x, max_y)
        x_axis_end = extrema["x_axis_end"]
        y_axis_end = extrema["y_axis_end"]
        x_ep = (origin[0], origin[1], x_axis_end[0], x_axis_end[1])
        y_ep = (origin[0], origin[1], y_axis_end[0], y_axis_end[1])

    # Length calculation requested:
    # Use per-endpoint depth at mask/bbox extrema points. The extrema-to-extrema segments
    # (e.g. min_x_point -> max_x_point) are generally NOT axis-aligned.
    # We therefore:
    #   1) compute metric length of that oblique segment using depths at both endpoints
    #   2) scale it by (axis_pixel_len / oblique_pixel_len) to approximate the length
    #      along the requested axes anchored at origin=(min_x, max_y).
    if extrema is None:
        x_len = None
        y_len = None
    else:
        # Axis pixel lengths (from bbox extents)
        axis_dx_px = float(extrema["max_x"] - extrema["min_x"])
        axis_dy_px = float(extrema["max_y"] - extrema["min_y"])

        # Oblique extrema segments inside mask
        min_x_point = extrema["min_x_point"]
        max_x_point = extrema["max_x_point"]
        min_y_point = extrema["min_y_point"]
        max_y_point = extrema["max_y_point"]

        oblique_dx_px = float(np.hypot(max_x_point[0] - min_x_point[0], max_x_point[1] - min_x_point[1]))
        oblique_dy_px = float(np.hypot(max_y_point[0] - min_y_point[0], max_y_point[1] - min_y_point[1]))

        x_oblique_m = _metric_distance_between_rgb_pixels(
            depth_img=depth_img,
            p1_rgb=min_x_point,
            p2_rgb=max_x_point,
            scale_factor=scale_factor,
            depth_unit_scale=float(depth_unit_scale),
            K=K,
        )
        y_oblique_m = _metric_distance_between_rgb_pixels(
            depth_img=depth_img,
            p1_rgb=min_y_point,
            p2_rgb=max_y_point,
            scale_factor=scale_factor,
            depth_unit_scale=float(depth_unit_scale),
            K=K,
        )

        if x_oblique_m is None or oblique_dx_px <= 0 or axis_dx_px <= 0:
            x_len = None
        else:
            x_len = float(x_oblique_m) * float(axis_dx_px / oblique_dx_px)

        if y_oblique_m is None or oblique_dy_px <= 0 or axis_dy_px <= 0:
            y_len = None
        else:
            y_len = float(y_oblique_m) * float(axis_dy_px / oblique_dy_px)

    # Save visuals
    # - main_mask.png: keep for debugging
    # - annotated_measure.jpg: requested to show ONLY the lines (no mask overlay)
    annotated = rgb_bgr.copy()
    if main_mask is not None:
        cv2.imwrite(str(output_dir / "main_mask.png"), (main_mask.astype(np.uint8) * 255))

    # Draw axis-aligned segments from origin
    if x_ep is not None:
        _draw_line(annotated, (x_ep[0], x_ep[1]), (x_ep[2], x_ep[3]), (0, 255, 0))
    if y_ep is not None:
        _draw_line(annotated, (y_ep[0], y_ep[1]), (y_ep[2], y_ep[3]), (255, 0, 0))

    # Draw guides using the 4 extremal points (inside mask) projected to axes (parallel guides)
    if extrema is not None:
        origin = extrema["origin"]
        # x guides: from max_x_point vertically to y=max_y
        max_x_point = extrema["max_x_point"]
        _draw_dashed_line(annotated, max_x_point, (max_x_point[0], origin[1]), (0, 255, 0), thickness=6)
        # y guides: from min_y_point horizontally to x=min_x
        min_y_point = extrema["min_y_point"]
        _draw_dashed_line(annotated, min_y_point, (origin[0], min_y_point[1]), (255, 0, 0), thickness=6)

    # Put total length labels on the image (requested)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    margin = 10

    if x_ep is not None and x_len is not None:
        mx = int(round((x_ep[0] + x_ep[2]) / 2))
        my = int(round((x_ep[1] + x_ep[3]) / 2))
        text = f"X: {x_len:.3f} {depth_unit_name}"
        (t_w, t_h), base_line = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Center horizontally, place below the line (outer side)
        # my is max_y (bottom edge), so add to y to go further down
        px = mx - t_w // 2
        py = my + t_h + margin + base_line

        _put_length_label(
            annotated,
            text=text,
            anchor_xy=(px, py),
            color=(0, 255, 0),
        )

    if y_ep is not None and y_len is not None:
        mx = int(round((y_ep[0] + y_ep[2]) / 2))
        my = int(round((y_ep[1] + y_ep[3]) / 2))
        text = f"Y: {y_len:.3f} {depth_unit_name}"
        (t_w, t_h), base_line = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Center vertically, place left of the line (outer side)
        # mx is min_x (left edge), so subtract from x to go further left
        px = mx - t_w - margin
        py = my + t_h // 2

        _put_length_label(
            annotated,
            text=text,
            anchor_xy=(px, py),
            color=(255, 0, 0),
        )

    cv2.imwrite(str(output_dir / "annotated_measure.jpg"), annotated)
    cv2.imwrite(str(axis_export_dir / f"{output_obj_id}.axis.jpg"), annotated)
    (output_dir / "caption.json").write_text(caption_json, encoding="utf-8")

    # Retrieve depths for the 4 extrema points used for calculation
    extrema_depths: Dict[str, Optional[float]] = {
        "min_x_point": None,
        "max_x_point": None,
        "min_y_point": None,
        "max_y_point": None,
    }

    if extrema is not None:
        for key in ["min_x_point", "max_x_point", "min_y_point", "max_y_point"]:
            pt = extrema[key]
            # pt is (x, y) in RGB coordinates
            d_raw = _depth_at_rgb_px(
                depth_img,
                rgb_u=float(pt[0]),
                rgb_v=float(pt[1]),
                scale_factor=scale_factor,
            )
            if d_raw is not None:
                extrema_depths[key] = float(d_raw) * float(depth_unit_scale)

    return MeasureResult(
        image=str(rgb_path),
        caption_json=caption_json,
        classes=classes,
        mask_strategy=mask_strategy,
        mask_shrink_px=int(used_shrink_px),
        x_endpoints_rgb=list(x_ep) if x_ep is not None else None,
        y_endpoints_rgb=list(y_ep) if y_ep is not None else None,
        x_length=x_len,
        y_length=y_len,
        depth_unit=depth_unit_name,
        depths=extrema_depths,
    )


def _default_paths() -> Dict[str, Path]:
    repo_root = Path(__file__).resolve().parent
    groundingdino_repo = repo_root / "GroundingDINO"
    checkpoint_dir = groundingdino_repo / "weights"
    return {
        "repo_root": repo_root,
        "groundingdino_repo": groundingdino_repo,
        "checkpoint_dir": checkpoint_dir,
    }


def main() -> None:
    load_dotenv()
    defaults = _default_paths()

    parser = argparse.ArgumentParser(
        description="VLM caption -> GroundingDINO+SAM mask -> measure longest X/Y extents with depth",
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default="Visphysquant/output.json",
        help="JSON with entries like {ID, images:[...]} (same as estimate_2dbbox_size.py)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of entries to process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(defaults["repo_root"] / "outputs" / "vlm_seg_measure"),
    )

    # VLM
    parser.add_argument("--vlm-provider", type=str, default="gpt", choices=["gpt", "gemini"])
    parser.add_argument("--vlm-model", type=str, default="gpt-5.2")
    parser.add_argument("--max-classes", type=int, default=8)
    parser.add_argument("--fixed-class", type=str, help="Skip VLM and use this class name directly (e.g. 'tong')")

    # GroundingDINO + SAM
    parser.add_argument("--groundingdino-repo", type=str, default=str(defaults["groundingdino_repo"]))
    parser.add_argument(
        "--dino-config",
        type=str,
        default=str(defaults["groundingdino_repo"] / "groundingdino/config/GroundingDINO_SwinT_OGC.py"),
    )
    parser.add_argument(
        "--dino-checkpoint",
        type=str,
        default=str(defaults["checkpoint_dir"] / "groundingdino_swint_ogc.pth"),
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default=str(defaults["checkpoint_dir"] / "sam_vit_h_4b8939.pth"),
    )
    parser.add_argument("--sam-encoder", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--mask-strategy",
        type=str,
        default="best_score",
        choices=["best_score", "union"],
    )
    parser.add_argument(
        "--box-select",
        type=str,
        default="best_score",
        choices=["best_score", "smallest"],
        help="When multiple bboxes are detected, choose which one to keep.",
    )
    parser.add_argument(
        "--mask-shrink-px",
        type=int,
        default=5,
        help="Erode the selected mask by this many pixels before measuring (each boundary shrinks inward).",
    )
    parser.add_argument(
        "--mask-shrink-ratio",
        type=float,
        default=0.0,
        help="Additional erosion amount as ratio of min(mask_height, mask_width). Used with --mask-shrink-px as max().",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )

    # Depth
    parser.add_argument(
        "--depth-unit-scale",
        type=float,
        default=1.0,
        help="Multiply raw depth values by this. For Record3D depth in mm: use 1.0 (output mm). For meters: use 0.001.",
    )
    parser.add_argument(
        "--depth-unit-name",
        type=str,
        default="mm",
        help="Label for output units (e.g. 'mm' or 'm')",
    )

    args = parser.parse_args()

    input_json = Path(args.input_json)
    if not input_json.exists():
        raise FileNotFoundError(f"input json not found: {input_json}")

    data = json.loads(input_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("input json must be a list")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for entry in data[: max(0, args.limit)]:
        obj_id = entry.get("ID")
        images = entry.get("images", [])
        if not images:
            continue

        rgb_path_str = _ensure_rgb_path(images[0])
        rgb_path = Path(rgb_path_str)
        if not rgb_path.exists():
            continue

        per_out = out_root / f"{_safe_filename(str(obj_id))}_{rgb_path.stem}"

        # Use class from JSON if available, otherwise fall back to CLI argument
        target_class = entry.get("class")
        if not target_class:
            target_class = args.fixed_class

        try:
            res = run_one(
                rgb_path=rgb_path,
                output_dir=per_out,
                groundingdino_repo=Path(args.groundingdino_repo),
                dino_config=Path(args.dino_config),
                dino_checkpoint=Path(args.dino_checkpoint),
                sam_checkpoint=Path(args.sam_checkpoint),
                sam_encoder=args.sam_encoder,
                device=args.device,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                mask_strategy=args.mask_strategy,
                vlm_provider=args.vlm_provider,
                vlm_model=args.vlm_model,
                max_classes=args.max_classes,
                depth_unit_scale=args.depth_unit_scale,
                depth_unit_name=args.depth_unit_name,
                mask_shrink_px=args.mask_shrink_px,
                mask_shrink_ratio=args.mask_shrink_ratio,
                fixed_class=target_class,
                box_select=args.box_select,
            )
            results.append({"ID": obj_id, **asdict(res)})
        except Exception as e:
            results.append({"ID": obj_id, "image": str(rgb_path), "error": str(e)})

    (out_root / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
