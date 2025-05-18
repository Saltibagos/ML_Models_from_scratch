import math

from SVM_Lin_Ker import *
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay


df=pd.read_csv('diabetes.csv')
X=df.drop(columns=['Outcome'],axis=1)
y=df['Outcome']

scaler=StandardScaler()
scaler.fit(X)
X_std=scaler.transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_std,y,test_size=0.2,random_state=42)

model=SVM(learning_rate=0.1,n_iters=1000,lambda_param=0.15)
model.fit(X_train,y_train)

train_prediction=model.predict(X_test)
training_accuracy=accuracy_score(y_test,train_prediction)

print(training_accuracy)


cm=confusion_matrix(y_test,train_prediction)
disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()





