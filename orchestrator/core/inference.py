# inference/inference.py

import os
import re
import base64
from io import BytesIO

import numpy as np
from PIL import Image as PILImage, ImageDraw
import torch
from torchvision import transforms
from scipy.ndimage import label

from .classes import LightningViTModel
from .functions import load_classdict

def get_bounding_boxes(binary_mask: np.ndarray):
    labeled_array, num_features = label(binary_mask)
    boxes = []
    for region_label in range(1, num_features + 1):
        coords = np.argwhere(labeled_array == region_label)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        boxes.append((y_min, x_min, y_max, x_max))
    return boxes

def run_segmentation_outputs(
    image_bytes: bytes,
    vision_model_id: int,
    base_path: str,
    classdict_csv: str,
    target_size: tuple[int,int] = (224,224),
) -> dict[str, bytes]:
    """
    Runs exactly as before, except takes a single `vision_model_id` instead of a list.
    Returns a dict { "mask.png": <PNG bytes>, "bbox.png": <PNG bytes> }.
    """

    rgb_to_class, class_names = load_classdict(classdict_csv)
    num_classes = len(class_names)

    # 1) Open image from bytes
    pil_img = PILImage.open(BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([transforms.Resize(target_size), transforms.ToTensor()])
    image_tensor = transform(pil_img).unsqueeze(0)  # [1,3,H,W]

    # 2) Fetch the single VisionModel config from your own DB or a local YAML.
    #    If you keep a mirror of VisionModel in ORCH_API or fetch it from VT_API, do so here.
    #    For simplicity, assume VT sent `patch_size, hidden_size, ...` directly, or you store a copy.
    #    Here we'll just show an example where you retrieve a local JSON or call VT_API to fetch it.

    #    (In most setups you'd replicate the minimal VisionModel table here, keyed by ID.)
    from .models import VisionModel  # or fetch from VT_API
    vm = VisionModel.objects.get(id=vision_model_id)
    patch_size = vm.patch_size
    hidden_size = vm.hidden_size
    hidden_layers = vm.num_hidden_layers
    attention_heads = vm.num_attention_heads

    # 3) Instantiate ViT model
    model = LightningViTModel(
        num_classes=num_classes,
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_hidden_layers=hidden_layers,
        num_attention_heads=attention_heads
    )

    # 4) Find latest checkpoint for this vision_model_id
    ckpt_dir = os.path.join(base_path, f"logs/vit-model/version_{vision_model_id}/checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt in {ckpt_dir}")

    def _step(fname: str) -> int:
        m = re.search(r"=([0-9]+)", fname)
        return int(m.group(1)) if m else -1

    latest = max(ckpt_files, key=_step)
    ckpt_path = os.path.join(ckpt_dir, latest)

    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt_data["state_dict"])
    model.eval()

    # 5) Forward
    with torch.no_grad():
        logits = model(image_tensor)             # [1, num_classes, H, W]
        pr_mask = logits.sigmoid().squeeze(0)    # [num_classes, H, W]
    pred_labels = pr_mask.argmax(dim=0).cpu().numpy()  # [H, W]

    # 6) Build colored mask
    index_to_color = np.zeros((num_classes, 3), dtype=np.uint8)
    for rgb, idx in rgb_to_class.items():
        index_to_color[idx] = list(rgb)

    colored_mask = index_to_color[pred_labels]  # [H, W, 3]
    mask_pil = PILImage.fromarray(colored_mask)
    buf_mask = BytesIO()
    mask_pil.save(buf_mask, format="PNG")
    buf_mask.seek(0)
    mask_png = buf_mask.read()
    buf_mask.close()

    # 7) Build bbox overlay
    target_img = pil_img.resize(target_size)
    draw = ImageDraw.Draw(target_img)
    for class_idx in range(1, num_classes):
        bin_mask = (pred_labels == class_idx).astype(np.uint8)
        if bin_mask.sum() == 0:
            continue
        boxes = get_bounding_boxes(bin_mask)
        rgb = tuple(int(c) for c in index_to_color[class_idx])
        for (y0, x0, y1, x1) in boxes:
            draw.rectangle([x0, y0, x1, y1], outline=rgb, width=2)

    buf_bbox = BytesIO()
    target_img.save(buf_bbox, format="PNG")
    buf_bbox.seek(0)
    bbox_png = buf_bbox.read()
    buf_bbox.close()

    return {"mask.png": mask_png, "bbox.png": bbox_png}
