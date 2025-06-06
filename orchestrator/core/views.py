from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from .tasks import process_job
from django.conf import settings

class EnqueueAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        if request.headers.get("X-ORCH-TOKEN") != settings.VT_API_TOKEN:
            return Response({"detail": "Forbidden"}, status=403)

        job_id   = request.data.get("job_id")
        model_id = request.data.get("vision_model_id")
        img_file = request.FILES.get("input_image")

        if not job_id or not model_id or img_file is None:
            return Response({"detail": "job_id, vision_model_id, input_image required"},
                            status=400)

        # read the bytes once; Celery will get them as an arg
        image_bytes = img_file.read()
        process_job.delay(job_id, int(model_id), image_bytes)   # ⬅ signature update
        return Response({"status": "enqueued"}, status=202)

