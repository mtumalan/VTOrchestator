from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from .tasks import process_job

class EnqueueAPIView(APIView):
    """
    Receives:
      POST /enqueue/
      {
        job_id: "<uuid>",
        vision_model_id: <int>,
        input_image_url: "<http://vt/.../file.png>"
      }
    Enqueues a Celery task and returns 202 Accepted.
    """
    permission_classes = [AllowAny]  # or check a shared secret header

    def post(self, request, *args, **kwargs):
        job_id = request.data.get("job_id")
        model_id = request.data.get("vision_model_id")
        img_url = request.data.get("input_image_url")

        if not job_id or not model_id or not img_url:
            return Response(
                {"detail": "job_id, vision_model_id, and input_image_url are required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 1) Enqueue the Celery task
        process_job.delay(job_id, model_id, img_url)
        return Response({"status": "enqueued"}, status=status.HTTP_202_ACCEPTED)
