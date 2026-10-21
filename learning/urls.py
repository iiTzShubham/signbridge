from django.urls import path

from . import views

app_name = "learning"

urlpatterns = [
    path("learn/", views.lesson_list, name="lesson_list"),
    path("learn/<slug:slug>/", views.lesson_detail, name="lesson_detail"),
    path("learn/<slug:slug>/practice/", views.practice, name="practice"),
    path("learn/<slug:slug>/complete/", views.complete, name="complete"),
]
