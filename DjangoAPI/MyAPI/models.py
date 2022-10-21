from django.db import models

# Create your models here.
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

    form_id = models.IntegerField(default=0)
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
    
    def __str__(self):
        return "{}, {}".format(self.form_id, self.age)