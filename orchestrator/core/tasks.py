import os
import requests
from django.conf import settings
from celery import shared_task

from .inference import run_segmentation_outputs


@shared_task(bind=True)
def process_job(self, job_id: str, vision_model_id: int, image_bytes: bytes):
    """
    Executed by Celery worker.

    Parameters
    ----------
    job_id : str
        UUID from VT.
    vision_model_id : int
        Primary-key of the VisionModel chosen by the user.
    image_bytes : bytes
        Raw JPG/PNG bytes that VT already sent in the enqueue request.

    Steps
    -----
    1) Run local inference (run_segmentation_outputs).
    2) PATCH VT /inference-jobs/{job_id}/orch_complete/ with:
         - status = DONE | FAILED
         - mask_image file (mask.png) if success
         - error_message if failure
    """
    # ─── 0. Quick sanity-check on the payload ─────────────────────
    if not image_bytes:
        _notify_vt(job_id, status="FAILED", error="Empty image payload")
        return

    # ─── 1. Classdict CSV path — identical to before ──────────────
    classdict_csv = os.path.join(
        settings.BASE_DIR,
        "VisionChallenge", "collaboration_it_mx",
        "output_images", "class_names_colors.csv"
    )
    if not os.path.exists(classdict_csv):
        _notify_vt(job_id, status="FAILED", error="Classdict CSV not found.")
        return

    # ─── 2. Run segmentation locally ──────────────────────────────
    try:
        outputs = run_segmentation_outputs(
            image_bytes=image_bytes,
            vision_model_id=vision_model_id,
            base_path=settings.BASE_DIR,
            classdict_csv=classdict_csv,
            target_size=(224, 224),
        )
    except Exception as exc:
        _notify_vt(job_id, status="FAILED", error=str(exc))
        return

    # ─── 3. Build multipart payload (mask.png) ────────────────────
    mask_bytes = outputs.get("mask.png")
    if mask_bytes is None:
        _notify_vt(job_id, status="FAILED", error="No mask generated.")
        return

    files = {"mask_image": ("mask.png", mask_bytes, "image/png")}
    _notify_vt(job_id, status="DONE", files=files)


# ──────────────────────────────────────────────────────────────────
# Helper: single place to POST / PATCH back to VT
# ──────────────────────────────────────────────────────────────────
def _notify_vt(job_id: str, *, status: str, error: str | None = None, files=None):
    """
    Send a PATCH to VT with consistent auth header.
    """
    patch_url = f"{settings.VT_API_URL}/inference-jobs/{job_id}/orch_complete/"
    data = {"status": status}
    if error:
        data["error_message"] = error

    headers = {"X-ORCH-TOKEN": settings.VT_API_TOKEN}

    try:
        requests.patch(patch_url, data=data, files=files, headers=headers, timeout=10)
    except Exception:
        # Last-chance logging; nothing else we can do.
        self = _notify_vt  # keep flake8 quiet for unused self if not inside task
        getattr(self, "logger", print)(
            f"[orchestrator] could not PATCH {patch_url}"
        )