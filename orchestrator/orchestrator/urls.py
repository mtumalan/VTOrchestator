from django.urls import path
from core.views import EnqueueAPIView

urlpatterns = [
    path("enqueue/", EnqueueAPIView.as_view(), name="enqueue"),
]
