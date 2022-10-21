
from django.contrib import admin
from django.urls import path, include
from django.conf.urls import include
from MyAPI import views

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('api/', MyAPI.site.urls),
    path('', include('MyAPI.urls')),

]
