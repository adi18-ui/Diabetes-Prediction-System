from django.shortcuts import render,redirect
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def home(request):
    return render(request, 'home.html')
   
def predict(request):
    return render(request,'predict.html')

def result(request):
    df = pd.read_csv("data/diabetes.csv")

    X=df.drop("Outcome", axis=1)
    Y=df["Outcome"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    model=LogisticRegression()
    model.fit(X_train, Y_train)

    val1=float(request.GET['n1'])
    val2=float(request.GET['n2'])
    val3=float(request.GET['n3'])
    val4=float(request.GET['n4'])
    val5=float(request.GET['n5'])
    val6=float(request.GET['n6'])
    val7=float(request.GET['n7'])
    val8=float(request.GET['n8'])
    arr1=[val1,val2,val3,val4,val5,val6,val7,val8]
    #from .ml_model import my_model
    arr2=np.array(arr1).reshape(1,-1)
    pred=model.predict(arr2)

    result2=""
    if pred==[1]:
        result1="Positive"
    else:
        result1="Negative"

    return render(request, "predict.html", {"result2":result1})
