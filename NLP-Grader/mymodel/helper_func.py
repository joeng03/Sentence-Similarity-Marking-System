from sentence_transformers import SentenceTransformer
import pandas as pd
import scipy
import re
import string 
import numpy as np
import math

model=SentenceTransformer(r'C:\Users\user\PROJECTS\semantic_search\mymodel\static\mymodel\roberta-base-nli-stsb-mean-tokens')

def collect_answers(data,i,answers,answer,split):
  for j in range(0,len(data[i][-1])):
    if(data[i][-1][j]=='/'):
      answer.append(data[i][-1][split:j])
      split=j+1
  if(split==0):
    answers.append(data[i][-1])
  else:
    answer.append(data[i][-1][split:j])
    answers.append(answer)
    answer=[]

def cosine_similarity(data,answers):
  minimum_similarity=0.8
  result=np.zeros([data.shape[0],data.shape[1]-1])
  for k in range(0,data.shape[0]):
    if(isinstance(answers[k],list)):
      sentence_embeddings=model.encode(answers[k])
    else:
      sentence_embeddings=model.encode([answers[k]])
    for l in range(0,data.shape[1]-1):
      if(type(data[k][l])==float):
        if not math.isnan(data[k][l]):
          query_embeddings = model.encode([data[k][l]])
          distance = 1-scipy.spatial.distance.cdist([query_embeddings[0]], sentence_embeddings, "cosine")[0]
          if (distance.max()>minimum_similarity):
            result[k][l]=1
      else:
        query_embeddings = model.encode([data[k][l]])
        distance = 1-scipy.spatial.distance.cdist([query_embeddings[0]], sentence_embeddings, "cosine")[0]
        if (distance.max()>minimum_similarity):
          result[k][l]=1
  return result   