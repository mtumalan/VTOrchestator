import os
import requests
from django.conf import settings
from celery import shared_task
from django.core.files.base import ContentFile

from .inference import run_segmentation_outputs


@shared_task(bind=True)
def process_job(self, job_id: str, vision_model_id: int, input_image_url: str):
    """
    1) Download image from VT_API
    2) Run inference (run_segmentation_outputs)
    3) Call VT_API PATCH /inference-jobs/{job_id}/orch_complete/ with mask_image file
    """

    # 1) Download the image bytes
    try:
        resp = requests.get(input_image_url, timeout=10)
        resp.raise_for_status()
        image_bytes = resp.content
    except Exception:
        # If we cannot fetch the image, inform VT_API that we failed
        patch_url = f"{settings.VT_API_URL}/inference-jobs/{job_id}/orch_complete/"
        data = {"status": "FAILED", "error_message": "Cannot fetch input image."}
        headers = {"Authorization": f"Token {settings.VT_API_TOKEN}"}
        requests.patch(patch_url, data=data, headers=headers, timeout=5)
        return

    # 2) Compute classdict CSV path
    classdict_csv = os.path.join(
        settings.BASE_DIR,
        "VisionChallenge", "collaboration_it_mx", "output_images", "class_names_colors.csv"
    )
    if not os.path.exists(classdict_csv):
        patch_url = f"{settings.VT_API_URL}/inference-jobs/{job_id}/orch_complete/"
        data = {"status": "FAILED", "error_message": "Classdict CSV not found."}
        headers = {"Authorization": f"Token {settings.VT_API_TOKEN}"}
        requests.patch(patch_url, data=data, headers=headers, timeout=5)
        return

    # 3) Run inference locally
    try:
        outputs = run_segmentation_outputs(
            image_bytes=image_bytes,
            vision_model_id=vision_model_id,
            base_path=settings.BASE_DIR,
            classdict_csv=classdict_csv,
            target_size=(224, 224),
        )
    except Exception as e:
        patch_url = f"{settings.VT_API_URL}/inference-jobs/{job_id}/orch_complete/"
        data = {"status": "FAILED", "error_message": str(e)}
        headers = {"Authorization": f"Token {settings.VT_API_TOKEN}"}
        requests.patch(patch_url, data=data, headers=headers, timeout=5)
        return

    # 4) Prepare a multipart/form-data payload with exactly one file (mask.png).
    #    If you want to send both mask.png and bbox.png, you could zip them or send two fields.
    mask_bytes = outputs.get("mask.png")
    if mask_bytes is None:
        patch_url = f"{settings.VT_API_URL}/inference-jobs/{job_id}/orch_complete/"
        data = {"status": "FAILED", "error_message": "No mask generated."}
        headers = {"Authorization": f"Token {settings.VT_API_TOKEN}"}
        requests.patch(patch_url, data=data, headers=headers, timeout=5)
        return

    files = {
        "mask_image": ("mask.png", mask_bytes, "image/png"),
    }
    data = {"status": "DONE"}  # no error_message

    # 5) Call back VT_API to set mask_image + status
    patch_url = f"{settings.VT_API_URL}/inference-jobs/{job_id}/orch_complete/"
    headers = {"Authorization": f"Token {settings.VT_API_TOKEN}"}

    try:
        requests.patch(patch_url, data=data, files=files, headers=headers, timeout=10)
    except Exception:
        # If callback fails, there’s not much we can do here. At best, log it.
        pass
