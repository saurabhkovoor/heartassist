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
    # email = models.CharField(max_length=100)

# class User(AbstractUser):
#     class Role(models.TextChoices):
#         ADMIN = "ADMIN", 'Admin'
#         PATIENT = "PATIENT", "Patient"
#         DOCTOR = "DOCTOR", "Doctor"
#     base_role = Role.ADMIN
#     role = models.CharField(max_length=50, choices=Role.choices)

#     def save(self, *args, **kwargs):
#         if not self.pk:
#             self.role = self.base_role
#             return super().save(*args, **kwargs)

# Create your models here.

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

# class DoctorManager(BaseUserManager):
#     def get_queryset(self, *args, **kwargs):
#         results = super().get_queryset(*args, **kwargs)
#         return results.filter(role=User.Role.DOCTOR)

# class Doctor(User):
#     base_role = User.Role.DOCTOR

#     doctor = DoctorManager()
#     class Meta:
#         proxy = True

# @receiver(post_save, sender=Doctor)
# def create_user_profile(sender, instance, created, **kwargs):
#     if created and instance.role == "DOCTOR":
#         DoctorProfile.objects.create(user=instance)

# class DoctorProfile(models.Model):
#     user = models.OneToOneField(User, on_delete=models.CASCADE)
#     doctor_id = models.IntegerField(null=True, blank=True)
    # doctor_id = models.CharField(max_length=50)
    # doctor_id = models.AutoField(primary_key=True)
    # doctor_id = models.BigAutoField(primary_key=True)
    # registrationNo = models.IntegerField()
    # placeOfPractice = models.CharField(max_length=15)
    # university = models.CharField(max_length=15)
    # phone = models.IntegerField()
    
    # password = models.CharField(max_length=15)
    # name = models.CharField(max_length=15)
    # email = models.CharField(max_length=15)

# class PatientManager(BaseUserManager):
#     def get_queryset(self, *args, **kwargs):
#         results = super().get_queryset(*args, **kwargs)
#         return results.filter(role=User.Role.PATIENT)

# class Patient(User):
#     base_role = User.Role.PATIENT

#     patient = PatientManager()

#     class Meta:
#         proxy = True

# class PatientProfile(models.Model):
#     user = models.OneToOneField(User, on_delete=models.CASCADE)
#     patient_id = models.IntegerField(null=True, blank=True)
    # patient_id = models.AutoField(primary_key=True)
    # connectedDoctor = models.ForeignKey(DoctorProfile, on_delete=models.DO_NOTHING)
    # password = models.CharField(max_length=15)
    # name = models.CharField(max_length=15)
    # email = models.CharField(max_length=15)
    # phone = models.IntegerField()

# @receiver(post_save, sender=Patient)
# def create_user_profile(sender, instance, created, **kwargs):
#     if created and instance.role == "PATIENT":
#         PatientProfile.objects.create(user=instance)

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

# class AdminManager(BaseUserManager):
#     def get_queryset(self, *args, **kwargs):
#         results = super().get_queryset(*args, **kwargs)
#         return results.filter(role=User.Role.ADMIN)

# class Admin(User):
#     base_role = User.Role.ADMIN

#     admin = AdminManager()

#     class Meta:
#         proxy = True

# @receiver(post_save, sender=Admin)
# def create_user_profile(sender, instance, created, **kwargs):
#     if created and instance.role == "ADMIN":
#         AdminProfile.objects.create(user=instance)

# class AdminProfile(models.Model):
#     user = models.OneToOneField(User, on_delete=models.CASCADE)
#     admin_id = models.IntegerField(null=True, blank=True)
    # admin_id = models.BigAutoField(primary_key=True)
    # admin_id = models.AutoField(primary_key=True)
    # password = models.CharField(max_length=15)
    # name = models.CharField(max_length=15)
    # email = models.CharField(max_length=15)
    # phone = models.IntegerField()