# from django.shortcuts import render # not needed


# Create your views here.
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
class SpaView(LoginRequiredMixin,TemplateView):
    template_name = "spa/index.html"
