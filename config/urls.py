from django.contrib import admin
from django.urls import include, path
from django.views.generic import TemplateView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("accounts.urls")),
    path("research/", include("research.urls")),
    path("recognize/", include("recognition.urls")),
    path("translate/", include("translation.urls")),
    path("", include("learning.urls")),
    path("", TemplateView.as_view(template_name="home.html"), name="home"),
]
