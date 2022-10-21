from django import forms
# from django.forms import ModelForm
# from . models import heartDiseasePrediction

class heartDiseasePredictionForm(forms.Form):
    form_id = forms.IntegerField()
    age = forms.IntegerField()
    sex = forms.ChoiceField(choices=[("Male", "Male"),("Female", "Female")])
    cp = forms.ChoiceField(choices=[("typical angina", "typical angina"), ("asymptomatic", "asymptomatic"), ("non-anginal", "non-anginal"), ("atypical angina", "atypical angina")])
    trestbps = forms.IntegerField()
    chol = forms.IntegerField()
    fbs = forms.ChoiceField(choices=[("TRUE", "TRUE"), ("FALSE", "FALSE")])
    restecg = forms.ChoiceField(choices=[("lv hypertrophy", "lv hypertrophy"), ("normal", "normal"), ("st-t abnormality", "st-t abnormality")])
    thalch = forms.IntegerField()
    exang = forms.ChoiceField(choices=[("TRUE", "TRUE"), ("FALSE", "FALSE")])
    oldpeak = forms.DecimalField(max_digits = 5, decimal_places=2)
    slope = forms.ChoiceField(choices=[("downsloping", "downsloping"), ("flat", "flat"), ("upsloping", "upsloping")])
    ca = forms.ChoiceField(choices=[("0", "0"), ("1", "1"), ("2", "2"), ("3", "3")])
    thal = forms.ChoiceField(choices=[("fixed defect", "fixed defect"), ("normal", "normal"), ("reversable defect", "reversable defect")])
    
# class MyForm(ModelForm):
# 	class Meta:
# 		model=heartDiseasePrediction
# 		fields = '__all__'
# 		#exclude = 'firstname'