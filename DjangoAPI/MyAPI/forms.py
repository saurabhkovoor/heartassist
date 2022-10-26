from django import forms
from django.forms import ModelForm
from . models import *

from django.contrib.auth.forms import UserCreationForm
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
    # fields = '__all__'
    #exclude = 'firstname'

class DoctorSignUpForm(UserCreationForm):
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=True)
    phone_number = forms.CharField(required=True)
    email = forms.CharField(required=True)
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
        user.save()
        doctor = Doctor.objects.create(user=user)
        doctor.phone_number=self.cleaned_data.get("phone_number")
        doctor.registrationNo = self.cleaned_data.get("registrationNo")
        doctor.placeOfPractice = self.cleaned_data.get("placeOfPractice")
        doctor.university = self.cleaned_data.get("university")
        doctor.save()
        return user

class PatientSignUpForm(UserCreationForm):
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=True)
    phone_number = forms.CharField(required=True)
    email = forms.CharField(required=True)
    connectedDoctor = forms.HiddenInput()

    class Meta(UserCreationForm.Meta):
        model = User
    
    @transaction.atomic
    def save(self):
        user = super().save(commit=False)
        user.is_patient = True
        user.first_name = self.cleaned_data.get('first_name')
        user.last_name = self.cleaned_data.get('last_name')
        user.save()
        patient = Patient.objects.create(user=user)
        patient.phone_number=self.cleaned_data.get("phone_number")
        patient.save()
        return user

class PatientSignUpForm(UserCreationForm):
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=True)
    phone_number = forms.CharField(required=True)
    email = forms.CharField(required=True)

    class Meta(UserCreationForm.Meta):
        model = User
    
    @transaction.atomic
    def save(self):
        user = super().save(commit=False)
        user.is_patient = True
        user.first_name = self.cleaned_data.get('first_name')
        user.last_name = self.cleaned_data.get('last_name')
        user.save()
        patient = Patient.objects.create(user=user)
        patient.phone_number=self.cleaned_data.get("phone_number")
        patient.save()
        return user