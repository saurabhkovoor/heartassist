
from django.contrib import admin
from django.urls import path, include
from django.conf.urls import include
from MyAPI import views
from DjangoAPI.spa.views import SpaView
from DjangoAPI.api.views import GreetingApi

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('api/', MyAPI.site.urls),
    # path('', include('MyAPI.urls')),
    path("accounts/", include("django.contrib.auth.urls")),
    path("api/greet", GreetingApi.as_view()),
    path("", SpaView.as_view(), name="spa"),
]
