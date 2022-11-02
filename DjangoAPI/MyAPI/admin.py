from django.contrib import admin

from . models import heartDiseasePrediction, User, Patient, Doctor
# Register your models here.
admin.site.register(heartDiseasePrediction)
admin.site.register(User)
admin.site.register(Patient)
admin.site.register(Doctor)

admin.site.site_header = "HeartAssist Administration"