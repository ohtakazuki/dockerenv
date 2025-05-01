from django.shortcuts import render
from django.http import HttpResponse
import datetime

def index(request):
  now = datetime.datetime.now()
  # タイムゾーンが適用されているか確認
  # from django.utils import timezone
  # now_aware = timezone.localtime(timezone.now())
  return HttpResponse(f'こんにちは！ただいまの日時は {now} です！')
