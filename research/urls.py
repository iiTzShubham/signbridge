from django.urls import path
from . import views

app_name = "research"
urlpatterns = [
    path("", views.comparison, name="comparison"),
    path("download/<str:artifact>/", views.download, name="download"),
]
