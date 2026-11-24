from django.urls import path
from . import views

app_name = "recognition"
urlpatterns = [
    path("", views.recognize, name="recognize"),
    path("letters/", views.record_letters, name="record_letters"),
]
