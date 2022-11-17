from django import forms
from django.forms import ModelForm
from . models import *

from django.contrib.auth import models
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django.db import transaction

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
            "fbs": "Fasting Blood Sugar (less than 120mg/dL)",
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

    def clean_trestbps(self, *args, **kwargs):
        trestbps = self.cleaned_data.get("trestbps")
        if trestbps < 80:
            raise forms.ValidationError("The resting blood pressure rate is too low. Choose values between 80 - 200 mmHg.")
        if trestbps > 200:
            raise forms.ValidationError("The resting blood pressure rate is too high. Choose values between 80 - 200 mmHg.")
        return trestbps

    def clean_chol(self, *args, **kwargs):
        chol = self.cleaned_data.get("chol")
        if chol < 85:
            raise forms.ValidationError("The cholestrol level is too low. Choose values between 85 - 610 mg/dl.")
        if chol > 610:
            raise forms.ValidationError("The cholestrol level is too high. Choose values between 85 - 610 mg/dl.")
        return chol
    
    def clean_thalch(self, *args, **kwargs):
        thalch = self.cleaned_data.get("thalch")
        if thalch < 60:
            raise forms.ValidationError("The maximum heart rate level is too low. Choose values between 60 - 210 bpm.")
        if thalch > 210:
            raise forms.ValidationError("The maximum heart rate level is too high. Choose values between 60 - 210 bpm.")
        return thalch

    def clean_oldpeak(self, *args, **kwargs):
        oldpeak = self.cleaned_data.get("oldpeak")
        if oldpeak < -10:
            raise forms.ValidationError("The ST depression level is too low. Choose values between -10.0 to 10.0.")
        if oldpeak > 10:
            raise forms.ValidationError("The ST depression level is too high. Choose values between -10.0 to 10.0.")
        return oldpeak

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
        patient.save()
        return user

class EditProfileForm(forms.ModelForm):
    class Meta:
        model = models.User
        fields = ["username","email", "first_name", "last_name"]
        
class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ["phone_number","image"]