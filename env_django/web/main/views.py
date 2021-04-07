from django.shortcuts import render
from django.http import HttpResponse
import datetime

def index(request):
  now = datetime.datetime.now()
  return HttpResponse(f'こんにちは！ただいまの日時は {now} です！')