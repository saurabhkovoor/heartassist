from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt

from rest_framework import viewsets
from rest_framework.decorators import api_view
from django.core import serializers
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
from django.http import JsonResponse
from django.http.response import JsonResponse
from django.contrib import messages
from rest_framework.parsers import JSONParser
from . models import User, heartDiseasePrediction, Patient, Doctor, Admin
from . forms import heartDiseasePredictionForm, PatientSignUpForm, DoctorSignUpForm, EditProfileForm,ProfileUpdateForm
from . serializers import heartDiseasePredictionSerializers

from django.core.files.storage import default_storage

from django.views.generic import CreateView
from django.contrib.auth import login, logout,authenticate, update_session_auth_hash
from django.contrib.auth.forms import AuthenticationForm, UserChangeForm, PasswordChangeForm
from django.contrib.auth.decorators import login_required

import pickle
import joblib
import json
import numpy as np
# from sklearn import preprocessing
import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict, Counter

from django.contrib.auth.decorators import login_required


# Create your views here.

@csrf_exempt
def heartDiseasePredictionAPI(request, id=0):
    if request.method=="GET":
        heartDiseasePred = heartDiseasePrediction.objects.all()
        heartDiseasePrediction_serializer = heartDiseasePredictionSerializers(heartDiseasePred, many=True)
        return JsonResponse(heartDiseasePrediction_serializer.data, safe=False)
    elif request.method == "POST":
        print(request)
        heartDiseasePrediction_data = JSONParser().parse(request)
        heartDiseasePrediction_serializer = heartDiseasePredictionSerializers(data = heartDiseasePrediction_data)
        if heartDiseasePrediction_serializer.is_valid():
            heartDiseasePrediction_serializer.save()
            return JsonResponse("Added Successfully", safe = False)
        return JsonResponse("Failed to Add", safe = False)
    elif request.method == "PUT":
        heartDiseasePrediction_data = JSONParser().parse(request)
        heartDiseasePred = heartDiseasePrediction.objects.get(id=heartDiseasePrediction_data["id"])
        heartDiseasePrediction_serializer=heartDiseasePredictionSerializers(heartDiseasePred,data=heartDiseasePrediction_data)
        if heartDiseasePrediction_serializer.is_valid():
            heartDiseasePrediction_serializer.save()
            return JsonResponse("Updated Successfully",safe=False)
        return JsonResponse("Failed to Update")
    elif request.method=='DELETE':
        heartDiseasePred=heartDiseasePrediction.objects.get(id=id)
        heartDiseasePred.delete()
        return JsonResponse("Deleted Successfully",safe=False)
    elif request.method =='PURGE':
        heartDiseasePred=heartDiseasePrediction.objects.all()
        heartDiseasePred.delete()
        return JsonResponse("Deleted All Successfully",safe=False)
    
class heartDiseasePredictionView(viewsets.ModelViewSet):
	queryset = heartDiseasePrediction.objects.all()
	serializer_class = heartDiseasePredictionSerializers

def ohevalue(df):
    ohe_col = ["age", "trestbps", "chol", "fbs", "thalch", "exang", "oldpeak", "ca", "sex_Female", "sex_Male", "cp_asymptomatic", "cp_atypical angina", "cp_non-anginal", "cp_typical angina", "restecg_lv hypertrophy", "restecg_normal", "restecg_st-t abnormality", "slope_downsloping", "slope_flat", "slope_upsloping", "thal_fixed defect", "thal_normal", "thal_reversable defect"]
    cat_columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    df_processed = pd.get_dummies(df, columns=cat_columns)
    newdict={}
    for i in ohe_col:
        if i in df_processed.columns:
            newdict[i]=df_processed[i].values
        else:
            newdict[i]=0
    newdf=pd.DataFrame(newdict)
    return newdf

def heartResult(request):
    try:
        mdl=joblib.load("MyAPI/heart-strat.pkl")
        scalers=joblib.load("MyAPI/scalers.pkl")
        X=scalers.transform(request)
        y_pred=mdl.predict(X)
        newdf=pd.DataFrame(y_pred, columns=['num'])
        return (newdf.values[0][0])
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)

def heartForm(request):
    if request.method == 'POST':
        form = heartDiseasePredictionForm(request.POST)
        if form.is_valid():
            age = form.cleaned_data['age']
            sex = form.cleaned_data['sex']
            cp = form.cleaned_data['cp']
            trestbps = form.cleaned_data['trestbps']
            chol = form.cleaned_data['chol']
            fbs = form.cleaned_data['fbs']
            restecg = form.cleaned_data['restecg']
            thalch = form.cleaned_data['thalch']
            exang = form.cleaned_data['exang']
            oldpeak = form.cleaned_data['oldpeak']
            slope = form.cleaned_data['slope']
            ca = form.cleaned_data['ca']
            thal = form.cleaned_data['thal']
            myDict = (request.POST).dict()
            df=pd.DataFrame(myDict, index=[0])
            
            addd = []
            if int(myDict["age"])>60:
                addd.append("age")
            if myDict["cp"]!="non-anginal":
                addd.append("cp")
            if int(myDict["trestbps"])>130:
                addd.append("trestbps")
            if int(myDict["chol"])>200:
                addd.append("chol")
            if myDict["fbs"]!="FALSE":
                addd.append("fbs")
            if myDict["restecg"]!="normal":
                addd.append("restecg")
            if int(myDict["thalch"])>160:
                addd.append("thalch")
            if myDict["exang"]!="TRUE":
                addd.append("exang")
            if int(myDict["oldpeak"]) < -4:
                addd.append("oldpeak")
            if myDict["slope"]!="upsloping":
                addd.append("slope")
            if int(myDict["ca"]) < 1:
                addd.append("ca")
            if myDict["thal"]!="normal":
                addd.append("thal")
           
            answer = heartResult(ohevalue(df))
            context = {"answer": answer, "form": form, "addd": addd}
            messages.success(request, "Application Status: {}".format(answer))
            request2 = request.POST.dict()
            request2["result"]= answer
            form = heartDiseasePredictionForm(request2)
            
            if request.user.is_authenticated:
                usero = get_object_or_404(User, username=request.user)
                userid = usero.id
                request2 = request.POST.dict()
                request2["result"]= answer
                form = heartDiseasePredictionForm(request2)
                form = form.save(commit=False)
                form.user = request.user
                form.save()
            else:
                form.save()
                print("not logged in")

            return render(request, "heartForm.html", context)
    else:
        form = heartDiseasePredictionForm()

    return render(request, "heartForm.html", {"form": form})

def register(request):
    return render(request, 'register.html')

class patient_register(CreateView):
    model = User
    form_class = PatientSignUpForm
    template_name = "patient_register.html"

    def form_valid(self, form):
        user = form.save()
        login(self.request, user)
        return redirect("/")

class doctor_register(CreateView):
    model = User
    form_class = DoctorSignUpForm
    template_name = "doctor_register.html"

    def form_valid(self, form):
        user = form.save()
        login(self.request, user)
        return redirect("/")

def login_request(request):
    if request.method=='POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None :
                login(request,user)
                return redirect('/')
            else:
                messages.error(request,"Invalid username or password")
        else:
            messages.error(request,"Invalid username or password")
    return render(request, 'login.html',
    context={'form':AuthenticationForm()})

def logout_view(request):
    logout(request)
    return redirect('/')

@login_required
def account(request):
    context = {}
    if request.user.is_doctor:
        print("this is a doctor")
        hasConnectedPatients = False
        heartForms = heartDiseasePrediction.objects.filter(user=request.user).order_by("-created_at")
        try:
            currentDoctor = Doctor.objects.get(user = request.user)
            connectedPatients = Patient.objects.filter(connectedDoctor=currentDoctor)
            print(connectedPatients)
            print("has connected patients")
            hasConnectedPatients = True
            context = {"connectedPatients": connectedPatients, "hasConnectedPatients": hasConnectedPatients, "heartForms": heartForms, "currentDoctor":currentDoctor}
            for cpat in connectedPatients:
                print(cpat.user.first_name)
        except:
            print("error")
            context = {}
        return render(request, 'account_doctor.html', context)
    elif request.user.is_patient:
        print("this is a patient")
        hasDoctor = False
        for hp in heartDiseasePrediction.objects.filter(user=request.user):
            print(hp.created_at.strftime("%d/%m/%Y, %H:%M:%S"))
        heartForms = heartDiseasePrediction.objects.filter(user=request.user).order_by("-created_at")
        
        try:
            print(Patient.objects.filter(user=request.user)[0].connectedDoctor)
            print("has doctor")
            hasDoctor = True
            availableDoctors=[]
        except:
            availableDoctors = Doctor.objects.filter()
            print("doesn't have doctor")

        if hasDoctor:
            context = {"hasDoctor": hasDoctor, 
            "connected_doctor":Patient.objects.get(user = request.user).connectedDoctor, 
            "heartForms": heartForms,
            "availableDoctors": availableDoctors}
        else:
            context = {"hasDoctor": hasDoctor, 
            "heartForms": heartForms,
            "availableDoctors": availableDoctors}
        return render(request, 'account_patient.html', context)
    else:
        print("this is a admin")
        return redirect('/admin/')


@login_required
def edit_account(request):
    if request.method =="POST":
        user_form = EditProfileForm(request.POST, instance=request.user)
        profile_form = ProfileUpdateForm(request.POST, request.FILES, instance=request.user)

        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            return redirect("/account/")
    else:
        user_form = EditProfileForm(instance=request.user)
        profile_form = ProfileUpdateForm(instance=request.user)
    context = {
        "user_form":user_form,
        "profile_form":profile_form
    }
    return render(request, "edit_account.html", context)

@login_required
def change_password(request):
    if request.method =="POST":
        form = PasswordChangeForm(data=request.POST, user = request.user)

        if form.is_valid():
            form.save()
            update_session_auth_hash(request,form.user)
            return redirect("/account/")
        
        else:
            return redirect("/change-password/")

    else:
        form = PasswordChangeForm(user=request.user)
        args={"form":form}
        return render(request, "change_password.html", args)

def change_connection(request, operation, pk):
    if operation == "add":
        #adding doctor
        p = Patient.objects.get(user = request.user)
        u = User.objects.get(pk = pk)
        d = Doctor.objects.get(user = u)
        p.connectedDoctor = d
        p.save()
    elif operation == "remove":
        p = Patient.objects.get(user = request.user)
        p.connectedDoctor = None
        p.save()
    elif operation == "remove-from-doctor":
        u = User.objects.get(pk = pk)
        p = Patient.objects.get(user = u)
        p.connectedDoctor = None
        p.save()
    elif operation == "view-trials":
        hasTrials = False
        u = User.objects.get(pk = pk)
        heartForms = heartDiseasePrediction.objects.filter(user=request.user).order_by("-created_at")
        heartForms2 = heartDiseasePrediction.objects.filter(user=u).order_by("-created_at")
        print(heartForms2)

        hasConnectedPatients = False
        try:
            currentDoctor = Doctor.objects.get(user = request.user)
            connectedPatients = Patient.objects.filter(connectedDoctor=currentDoctor)
            hasTrials = True
            print(connectedPatients)
            print("has connected patients")
            hasConnectedPatients = True
            context={"heartForms2": heartForms2, "hasTrials": hasTrials,"connectedPatients": connectedPatients, "hasConnectedPatients": hasConnectedPatients, "heartForms": heartForms, "currentDoctor":currentDoctor, "selectedPatient":u}
            for cpat in connectedPatients:
                print(cpat.user.first_name)
        except:
            print("error")
            context = {}            
        return render(request, 'account_doctor.html', context)
    return redirect("/account/")