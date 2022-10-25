from django.urls import path, include, re_path
# from django.conf.urls import url

# from django.conf.urls import re_path
from django.conf.urls.static import static
from django.conf import settings

from . import views
from rest_framework import routers

router = routers.DefaultRouter()
router.register('MyAPI', views.heartDiseasePredictionView)
urlpatterns = [
	path('form/', views.cxcontact, name='cxform'),
    path('api/', include(router.urls)),
    path('api-auth/', include("rest_framework.urls", namespace="rest_framework")),
    path('status/', views.heartResult),
    path('register/', views.register,name='register'),
    path('patient_register/', views.patient_register.as_view(), name='patient_register'),
    path('doctor_register/', views.doctor_register.as_view(), name='doctor_register'),
    re_path(r'^heartDiseasePrediction$',views.heartDiseasePredictionAPI),
    re_path(r'^heartDiseasePrediction/([0-9]+)$',views.heartDiseasePredictionAPI),

    # re_path(r'^heartDiseasePrediction/savefile',views.SaveFile)
]+static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
