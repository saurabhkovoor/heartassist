from rest_framework import serializers
from . models import heartDiseasePrediction, Patient, Doctor, Admin

class heartDiseasePredictionSerializers(serializers.ModelSerializer):
    class Meta:
        model = heartDiseasePrediction
        # fields = '__all__'
        fields=("id","age","sex","cp","trestbps","chol","fbs","restecg","thalch","exang","oldpeak","slope","ca","thal","created_at","patient","result") 

# class patientSerializers(serializers.ModelSerializer):
#     class Meta:
#         model = Patient
#         fields=("phone","connectedDoctor")


# class doctorSerializers(serializers.ModelSerializer):
#     class Meta:
#         model = Doctor
#         fields=("phone","registrationNo","placeOfPractice","university")
   
# class adminSerializers(serializers.ModelSerializer):
#     class Meta:
#         model = Admin
#         fields=("phone")
