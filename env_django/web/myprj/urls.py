from django.contrib import admin
from django.urls import path, include # include をインポート

urlpatterns = [
  path('', include('main.urls')), # main アプリの URL をインクルード
  path('admin/', admin.site.urls),
]
