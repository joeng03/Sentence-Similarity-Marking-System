from django.urls import path
from django.conf.urls import url
from . import views

app_name='mymodel'
urlpatterns=[
    url('',views.upload_csv,name='upload_csv')]
