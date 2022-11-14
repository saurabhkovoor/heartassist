from asyncio import AbstractServer
from enum import unique
from django.db import models
# from djongo import models
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db.models.signals import post_save
from django.dispatch import receiver

class User(AbstractUser):
    is_doctor = models.BooleanField(default=False)
    is_patient = models.BooleanField(default=False)
    is_admin = models.BooleanField(default = False)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    phone_number = models.CharField(max_length=20, default="0123456789")
    image = models.ImageField(upload_to="profile_image", blank=True, default="profile_image/profile.png")

class Doctor(models.Model):
    user = models.OneToOneField(User, on_delete = models.CASCADE, primary_key = True)
    registrationNo = models.CharField(max_length=15)
    placeOfPractice = models.CharField(max_length=15)
    university = models.CharField(max_length = 15)

class Patient(models.Model):
    user = models.OneToOneField(User, on_delete = models.CASCADE, primary_key = True)
    connectedDoctor = models.ForeignKey(Doctor, on_delete=models.DO_NOTHING)

class Admin(models.Model):
    user = models.OneToOneField(User, on_delete = models.CASCADE, primary_key = True)

class heartDiseasePrediction(models.Model):
    SEX_CHOICES = (
        ("Male", "Male"),
        ("Female", "Female")
    )
    
    CP_CHOICES = (
        ("typical angina", "typical angina"),
        ("asymptomatic", "asymptomatic"),
        ("non-anginal", "non-anginal"),
        ("atypical angina", "atypical angina")
    )

    FBS_CHOICES = (
        ("TRUE", "TRUE"),
        ("FALSE", "FALSE")
    )

    RESTECG_CHOICES = (
        ("lv hypertrophy", "lv hypertrophy"),
        ("normal", "normal"),
        ("st-t abnormality", "st-t abnormality")
    )

    EXANG_CHOICES = (
        ("TRUE", "TRUE"),
        ("FALSE", "FALSE")
    )

    SLOPE_CHOICES = (
        ("downsloping", "downsloping"),
        ("flat", "flat"),
        ("upsloping", "upsloping")
    )

    CA_CHOICES = (
        ("0", "0"),
        ("1", "1"),
        ("2", "2"),
        ("3", "3")
    )

    THAL_CHOICES = (
        ("fixed defect", "fixed defect"),
        ("normal", "normal"),
        ("reversable defect", "reversable defect")
    )

    id = models.AutoField(primary_key=True)
    age = models.IntegerField(default=0)
    sex = models.CharField(max_length=15, choices=SEX_CHOICES)
    cp = models.CharField(max_length=15, choices=CP_CHOICES)
    trestbps = models.IntegerField(default=0)
    chol = models.IntegerField(default=0)
    fbs = models.CharField(max_length=15, choices=FBS_CHOICES)
    restecg = models.CharField(max_length=30, choices=RESTECG_CHOICES)
    thalch = models.IntegerField(default=0)
    exang = models.CharField(max_length=15, choices=EXANG_CHOICES)
    oldpeak = models.DecimalField(max_digits = 5, decimal_places=2)
    slope = models.CharField(max_length=15, choices=SLOPE_CHOICES)
    ca = models.CharField(max_length=15, choices=CA_CHOICES)
    thal = models.CharField(max_length=30, choices=THAL_CHOICES)
    user = models.ForeignKey(User, on_delete=models.CASCADE, default= None, blank = True)
    created_at = models.DateTimeField('%d/%m/%Y %H:%M', auto_now_add=True)
    result = models.IntegerField(default=0)
    # def __str__(self):
    #     return "{}, {}".format(self.id, self.age)