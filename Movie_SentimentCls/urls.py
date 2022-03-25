from django.contrib import admin
from django.urls import path
from django.urls import re_path
from Movie_SentimentCls import views

urlpatterns = [
    path('admin/', admin.site.urls),
    re_path('^$', views.index, name='Homepage'),
    re_path('predictRev', views.predictRev, name='PredictRev'),
    # re_path('viewDataBase', views.viewDatabase, name='viewDatabase'),
    # re_path('updateDataBase', views.updateDataBase, name='updateDataBase'),

]