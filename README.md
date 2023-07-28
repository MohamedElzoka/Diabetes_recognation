# Diabetes_recognation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as model_selection
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.metrics import classification_report

plt.style.use('fivethirtyeight')

import warnings
#warnings.filterwarnings('ignore')
data=pd.read_csv('/content/diabetes.csv')
data.head()
data.info()
data.describe()
data.duplicated().sum()
data.corr()
sns.heatmap(data.corr(), annot=True,fmt='0.1f',linewidths=.5)
sns.countplot(x='Outcome',data=data,palette=['b','r']
plt.figure(figsize=(20,6))
plt.subplot(1,3,1)
plt.title('counter plot')
sns.countplot(x='Pregnancies',data=data)

plt.subplot(1,3,2)
plt.title('distribution plot')
sns.displot(data["Pregnancies"])
plt.subplot(1,3,3)
plt.title('Box Plot')
sns.boxplot(y=data["Pregnancies"])
plt.show()
sns.boxplot(data.Age)
x=data.drop('Outcome',axis=1)
y=data['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)
model1=LogisticRegression()
model2=SVC()
model3=RandomForestClassifier()
model4=GradientBoostingClassifier(n_estimators=1000)
model5=KNeighborsClassifier()
model6=GaussianNB()
col=['LogisticRegression','SVC','RandomForestClassifier','GradientBoostingClassifier',' KNeighborsClassifier','GaussianNB']
result1=[]
result2=[]
result3=[]
def cal(model):
    model.fit(x_train,y_train) 
    pre=model.predict(x_test)
    acc=accuracy_score(pre,y_test)
    re=recall_score(pre,y_test)
    f1=f1_score(pre,y_test)
    result1.append(acc)
    result2.append(re)
    result3.append(f1)
    confusion_matrix(pre,y_test)
    sns.heatmap(confusion_matrix(pre,y_test),annot=True)
    print(model)
    print('accuracy=',acc,'recall=',re,'f1=',f1)
cal(model1)
cal(model2)
cal(model3)
cal(model4)
cal(model5)
cal(model6)
final_result = pd.DataFrame({'Algorithms':col, 'Accuracies': result1, 'Recall': result2, 'F1_Score': result3})

print(final_result)
fig,ax=plt.subplots(figsize=(15,5))
plt.plot(final_result.Algorithms,result1, label='Accuracies')
plt.plot(final_result.Algorithms,result2 , label='Recall')
plt.plot(final_result.Algorithms,result3 , label='Fi_Score')
plt.legend()
plt.show()

