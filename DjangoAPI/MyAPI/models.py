from enum import unique
from django.db import models

# Create your models here.
class Doctor(models.Model):
    doctor_id = models.AutoField(primary_key=True)
    password = models.CharField(max_length=15)
    name = models.CharField(max_length=15)
    registrationNo = models.IntegerField()
    placeOfPractice = models.CharField(max_length=15)
    university = models.CharField(max_length=15)
    email = models.CharField(max_length=15)
    phone = models.IntegerField()

class Patient(models.Model):
    patient_id = models.AutoField(primary_key=True)
    password = models.CharField(max_length=15)
    name = models.CharField(max_length=15)
    email = models.CharField(max_length=15)
    phone = models.IntegerField()
    connectedDoctor = models.ForeignKey(Doctor, on_delete=models.DO_NOTHING)

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
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, default= None)
    created_at = models.DateTimeField('%m/%d/%Y %H:%M:%S', auto_now_add=True)
    def __str__(self):
        return "{}, {}".format(self.id, self.age)

class Admin(models.Model):
    admin_id = models.AutoField(primary_key=True)
    password = models.CharField(max_length=15)
    name = models.CharField(max_length=15)
    email = models.CharField(max_length=15)
    phone = models.IntegerField()