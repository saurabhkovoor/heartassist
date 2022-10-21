from rest_framework import serializers
from . models import heartDiseasePrediction

class heartDiseasePredictionSerializers(serializers.ModelSerializer):
    class Meta:
        model = heartDiseasePrediction
        fields = '__all__'
