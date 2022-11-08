from django import forms
from django.forms import ModelForm
from . models import *

from django.contrib.auth import models #might not need
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django.db import transaction

# class heartDiseasePredictionForm(forms.Form):
#     form_id = forms.HiddenInput()
#     age = forms.IntegerField(widget=forms.TextInput(
#         attrs={'placeholder': "Enter your age"}))
#     sex = forms.ChoiceField(choices=[("Male", "Male"), ("Female", "Female")])
#     cp = forms.ChoiceField(choices=[("typical angina", "typical angina"), ("asymptomatic",
#                            "asymptomatic"), ("non-anginal", "non-anginal"), ("atypical angina", "atypical angina")])
#     trestbps = forms.IntegerField()
#     chol = forms.IntegerField()
#     fbs = forms.ChoiceField(choices=[("TRUE", "TRUE"), ("FALSE", "FALSE")])
#     restecg = forms.ChoiceField(choices=[("lv hypertrophy", "lv hypertrophy"), (
#         "normal", "normal"), ("st-t abnormality", "st-t abnormality")])
#     thalch = forms.IntegerField()
#     exang = forms.ChoiceField(choices=[("TRUE", "TRUE"), ("FALSE", "FALSE")])
#     oldpeak = forms.DecimalField(max_digits=5, decimal_places=2)
#     slope = forms.ChoiceField(choices=[(
#         "downsloping", "downsloping"), ("flat", "flat"), ("upsloping", "upsloping")])
#     ca = forms.ChoiceField(
#         choices=[("0", "0"), ("1", "1"), ("2", "2"), ("3", "3")])
#     thal = forms.ChoiceField(choices=[("fixed defect", "fixed defect"), (
#         "normal", "normal"), ("reversable defect", "reversable defect")])
#     user = forms.HiddenInput()
#     date_time = forms.HiddenInput()
#     result = forms.HiddenInput()

class heartDiseasePredictionForm(ModelForm):
    class Meta:
        model = heartDiseasePrediction
        fields = ("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg","thalch", "exang", "oldpeak", "slope", "ca", "thal","result","user",)
        widgets={
            "result": forms.HiddenInput(),
            "user": forms.HiddenInput()
        }
        labels={
            "sex": "Gender",
            "cp": "Chest Pain Type",
            "trestbps": "Resting Blood Pressure (mmHg)",
            "chol": "Cholesterol (mg/dL)",
            "fbs": "Fasting Blood Sugar (more than 120mg/dL)",
            "restecg": "Resting ECG Results",
            "thalch": "Maximum Heart Rate (beats per min)",
            "exang": "Presence of Exercise Induced Angina",
            "oldpeak": "ST Depression",
            "slope": "Slope of Peak Exercise ST Segment (-10.0 to 10.0)",
            "ca": "No. of Fluoroscopy-Coloured Major Vessels",
            "thal": "Thalium Stress Test"
        }
    # fields = '__all__'
    #exclude = 'firstname'
    def clean_age(self, *args, **kwargs):
        age = self.cleaned_data.get("age")
        if age < 0:
            raise forms.ValidationError("Age cannot be negative.")
        if age < 27:
            raise forms.ValidationError("You are too young. This model best caters to patients older than 27")
        return age

class DoctorSignUpForm(UserCreationForm):
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=True)
    phone_number = forms.CharField(required=True)
    email = forms.EmailField(required=True)
    registrationNo = forms.CharField(required=True)
    placeOfPractice = forms.CharField(required=True)
    university = forms.CharField(required=True)

    class Meta(UserCreationForm.Meta):
        model = User

    @transaction.atomic
    def save(self):
        user = super().save(commit=False)
        user.is_doctor = True
        user.first_name = self.cleaned_data.get('first_name')
        user.last_name = self.cleaned_data.get('last_name')
        user.email = self.cleaned_data.get("email")
        user.phone_number = self.cleaned_data.get("phone_number")
        user.save()
        doctor = Doctor.objects.create(user=user)
        # doctor.phone_number=self.cleaned_data.get("phone_number")
        doctor.registrationNo = self.cleaned_data.get("registrationNo")
        doctor.placeOfPractice = self.cleaned_data.get("placeOfPractice")
        doctor.university = self.cleaned_data.get("university")
        doctor.save()
        return user

class PatientSignUpForm(UserCreationForm):
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=True)
    phone_number = forms.CharField(required=True)
    email = forms.EmailField(required=True)
    connectedDoctor = forms.HiddenInput()

    class Meta(UserCreationForm.Meta):
        model = User
    
    @transaction.atomic
    def save(self):
        user = super().save(commit=False)
        user.is_patient = True
        user.first_name = self.cleaned_data.get('first_name')
        user.last_name = self.cleaned_data.get('last_name')
        user.email = self.cleaned_data.get("email")
        user.phone_number = self.cleaned_data.get("phone_number")
        user.save()
        patient = Patient.objects.create(user=user)
        # patient.phone_number=self.cleaned_data.get("phone_number")
        patient.save()
        return user

# class PatientSignUpForm(UserCreationForm):
#     first_name = forms.CharField(required=True)
#     last_name = forms.CharField(required=True)
#     phone_number = forms.CharField(required=True)
#     email = forms.CharField(required=True)

#     class Meta(UserCreationForm.Meta):
#         model = User
    
#     @transaction.atomic
#     def save(self):
#         user = super().save(commit=False)
#         user.is_patient = True
#         user.first_name = self.cleaned_data.get('first_name')
#         user.last_name = self.cleaned_data.get('last_name')
#         user.save()
#         patient = Patient.objects.create(user=user)
#         patient.phone_number=self.cleaned_data.get("phone_number")
#         patient.save()
#         return user

# class EditProfileForm(UserChangeForm):
#     class Meta:
#         model = models.User
#         fields = (
#             "email",
#             "first_name",
#             "last_name"
#         )
#         exclude=()

class EditProfileForm(forms.ModelForm):
    class Meta:
        model = models.User
        fields = ["username","email", "first_name", "last_name"]
        # widgets = {
        #     'password': forms.PasswordInput()
        # }
class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ["phone_number","image"]