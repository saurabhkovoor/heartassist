from django.urls import path, include, re_path
# from django.conf.urls import url

# from django.conf.urls import re_path
from django.conf.urls.static import static
from django.conf import settings

from . import views
from rest_framework import routers
from django.contrib.auth.views import PasswordResetView, PasswordResetDoneView, PasswordResetConfirmView,PasswordResetCompleteView

router = routers.DefaultRouter()
router.register('MyAPI', views.heartDiseasePredictionView)
urlpatterns = [
	# path('form/', views.heartForm, name='heartForm'),
    path("", views.heartForm, name="heartForm"),
    # path('', views.heartForm, name='Homepage'),
    path('api/', include(router.urls)),
    path('api-auth/', include("rest_framework.urls", namespace="rest_framework")),
    path('status/', views.heartResult),
    path('register/', views.register,name='register'),
    path('patient_register/', views.patient_register.as_view(), name='patient_register'),
    path('doctor_register/', views.doctor_register.as_view(), name='doctor_register'),
    path('doctor_register/', views.doctor_register.as_view(), name='doctor_register'),
    path('login/', views.login_request, name="login"),
    path('logout/', views.logout_view, name="logout"),
    path('account/', views.account, name = 'account'),
    re_path(r"account/edit/$", views.edit_account2, name="edit_account"),
    re_path(r"change-password/$", views.change_password, name="change_password"),
    re_path(r"reset-password/$", PasswordResetView.as_view(), name="reset_password"),
    re_path(r"reset-password/done/$", PasswordResetDoneView.as_view(), name="password_reset_done"),
    re_path(r"reset-password/confirm/(?P<uidb64>[0-9A-Za-z]+)-(?P<token>.+)/$", PasswordResetConfirmView.as_view(), name="password_reset_confirm"),
    re_path(r"reset-password/complete/$", PasswordResetCompleteView.as_view(), name="password_reset_complete"),
    re_path(r"connect/(?P<operation>.+)/(?P<pk>\d+)/$", views.change_connection, name="change_connection"),
    re_path(r'^heartDiseasePrediction$',views.heartDiseasePredictionAPI),
    re_path(r'^heartDiseasePrediction/([0-9]+)$',views.heartDiseasePredictionAPI),

    # re_path(r'^heartDiseasePrediction/savefile',views.SaveFile)
]+static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
