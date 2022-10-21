from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import api_view
from django.core import serializers
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
from django.http import JsonResponse
from django.contrib import messages
from rest_framework.parsers import JSONParser
from . models import heartDiseasePrediction
from . forms import heartDiseasePredictionForm
from . serializers import heartDiseasePredictionSerializers
import pickle
import joblib
import json
import numpy as np
# from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict, Counter

# Create your views here.

class heartDiseasePredictionView(viewsets.ModelViewSet):
	queryset = heartDiseasePrediction.objects.all()
	serializer_class = heartDiseasePredictionSerializers

# def myform(request):
#     if request.method=="POST":
#         form = MyForm(request.POST)
#         if form.is_valid():
#             myform = form.save(commit=False)
#             # myform.save()
#     else:
#         form = MyForm()
#     return render(request, 'myform/form.html', {'form': form})
# @api_view(["POST"])

def ohevalue(df):
    # ohe_col=joblib.load("MyAPI/heart-strat2.pkl")
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
        mdl=joblib.load("MyAPI/heart-strat2.pkl")
        # mydata=request.data
        # unit=np.array(list(mydata.values()))
        # unit=unit.reshape(1,-1)
        scalers=joblib.load("MyAPI/scalers.pkl")
        # scalers = MinMaxScaler()
        X=scalers.transform(request)
        y_pred=mdl.predict(X)
        newdf=pd.DataFrame(y_pred, columns=['num'])
        # newdf=newdf.replace({0:'0', 1:'1', 2:'2', 3:'3'})
        # print('Your Status is {}'.format(newdf))
        # return ("Your Status is {}".format(newdf))
        return (newdf.values[0][0])
        # return JsonResponse('Your Status is {}'.format(newdf), safe=False)
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)
    #     return Response(e.args[0])
        # return "hello"


def cxcontact(request):
    if request.method == 'POST':
        form = heartDiseasePredictionForm(request.POST)
        if form.is_valid():
            form_id = form.cleaned_data['form_id']
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
            # print(myDict)
            df=pd.DataFrame(myDict, index=[0])
            
            # answer=heartResult(ohevalue(df))[0]
            # Xscalers=heartResult(ohevalue(df))[1]
            # print(Xscalers)
            # messages.success(request,'Application Status: {}'.format(answer))
            # print(df)
            # print(ohevalue(df))
            answer = heartResult(ohevalue(df))
            messages.success(request, "Application Status: {}".format(answer))
            # print(ohevalue(df))
    form = heartDiseasePredictionForm()

    return render(request, "myform/cxform.html", {"form": form})
            