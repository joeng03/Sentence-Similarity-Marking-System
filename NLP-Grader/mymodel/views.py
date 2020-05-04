from django.shortcuts import render 
from django.http import HttpResponse,HttpResponseRedirect
from django.contrib import messages 
from django.urls import reverse 
import pandas as pd
import scipy
import numpy as np
import math
from sentence_transformers import SentenceTransformer
from django.views.static import serve
from .helper_func import collect_answers,cosine_similarity
import os

def upload_csv(request):
	data={}
	if "GET" == request.method:
		return render(request, "mymodel/index.html", data)
	try:
		csv_file = request.FILES["csv_file"]
		if not csv_file.name.endswith('.csv'):
			messages.error(request,'File is not CSV type')
			return render(request,'mymodel/error.html',{'filename':csv_file.name})
        #if file is too large, return
		if csv_file.multiple_chunks():
			messages.error(request,"Uploaded file is too big (%.2f MB)." % (csv_file.size/(1000*1000),))
			return HttpResponseRedirect(reverse("mymodel:upload_csv"))
		
	except Exception as e:
		messages.error(request,"Unable to upload file. "+repr(e))
	LIST=[]
	df=pd.read_csv(csv_file)
	for index in range(1,df.shape[1]):
		LIST.append(df.iloc[: ,index])
	data=np.array(LIST)
	answers=[]
	answer=[]
	for i in range(0,data.shape[0]):
		split=0
		if type(data[i][-1])==float:
			if not math.isnan(data[i][-1]):
				collect_answers(data,i,answers,answer,split)
			else:
				answers.append('kosong')
		else:
			collect_answers(data,i,answers,answer,split)
	results=cosine_similarity(data,answers)
	marks=[]
	for result in results.T:
		marks.append(result.sum())
	marks.append('---')
	df["Total marks"]=marks
	response = HttpResponse(content_type='text/csv')
	response['Content-Disposition'] = 'attachment; filename="new_csv.csv"'
	df.to_csv(path_or_buf=response,index=False)
	return response










