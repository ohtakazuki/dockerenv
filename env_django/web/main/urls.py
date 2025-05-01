from django.urls import path
from . import views

app_name = 'main' # アプリケーションの名前空間 (推奨)
urlpatterns = [
  path('', views.index, name='index')
]
