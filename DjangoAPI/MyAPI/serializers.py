from rest_framework import serializers
from . models import heartDiseasePrediction, Patient, Doctor, Admin

class heartDiseasePredictionSerializers(serializers.ModelSerializer):
    class Meta:
        model = heartDiseasePrediction
        # fields = '__all__'
        fields=("id","age","sex","cp","trestbps","chol","fbs","restecg","thalch","exang","oldpeak","slope","ca","thal","created_at","user","result") 
