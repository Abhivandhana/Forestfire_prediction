import pandas as pd
from google.colab import files
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
uploaded = files.upload()
data = pd.read_excel(uploaded["Forest_fire.xlsx"])
print('Data before Transformation \n',data)
le = LabelEncoder()
data['Fire'] = le.fit_transform(data['Fire'])
print('Data after Transformation \n',data)
inplist = data.columns[:-1]
print('Data before Scaling \n',data)
scale = StandardScaler()
data[inplist] = scale.fit_transform(data[inplist])
print('Data after Scaling \n',data)
x = data.values[:,:-1]
y = data.values[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=11)
clf = MLPClassifier(hidden_layer_sizes=(3,),activation="logistic",max_iter=150,solver='adam',learning_rate='constant',learning_rate_init=0.19)
clf.fit(x_train,y_train)
ypred = clf.predict(x_test)
cm = confusion_matrix(y_test,ypred)
print('Çonfusion Matrix \n',cm)
print('Classification Report \n',classification_report(y_test,ypred))
print('Coefficients',clf.coefs_)
print('Intercepts',clf.intercepts_)
loss_values = clf.loss_curve_
plt.plot(loss_values)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
