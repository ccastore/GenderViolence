import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
import pandas as pd
import seaborn as sn
import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow import keras
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

#Data and clas import
dir_data = 'Data.txt'
dir_clas = 'Clases.txt'
data = pd.read_csv(dir_data, header=None)
clas = pd.read_csv(dir_clas, header=None)

#---------------------------------------------------------------------------
#Training and test data
test_size=0.40
x_train,x_test,y_train,y_test = train_test_split(data, clas, test_size=test_size)

#---------------------------------------------------------------------------
#Matrix converted
ngram_range=(1,3)
min_df=50

vectorizer = CountVectorizer(analyzer='word',min_df=min_df,ngram_range=ngram_range)
#vectorizer = HashingVectorizer(n_features=350,analyzer='word',ngram_range=ngram_range)
#vectorizer = TfidfVectorizer(analyzer='word',min_df=min_df,ngram_range=ngram_range)

X_train = vectorizer.fit_transform(x_train[0][:])
X_train=X_train.toarray()
min_max_scaler = preprocessing.MinMaxScaler()
x1 = min_max_scaler.fit_transform(X_train)


#Sampling Metod
#--------------------------------------------------------------------------
#metodoTraining = SMOTE(n_jobs=7)
metodoTraining = RandomOverSampler(random_state=42)
x1, y1 = metodoTraining.fit_resample(x1, y_train[0][:])
X_test=vectorizer.transform(x_test[0][:])
X_test=X_test.toarray()
x2 = min_max_scaler.transform(X_test)
y2 = np.array(y_test[0][:])

#NeuralNetwork
epochs=20
batch_size=150
tf.keras.backend.clear_session()
#-----------------------------------------------------------------------------------------------
#Model generation
model = Sequential()
model.add(Dense(6,input_shape=(x1.shape[1],),activation='sigmoid'))
model.add(Dense(6,activation='sigmoid'))
model.add(Dense(5,activation='sigmoid'))
model.add(Dense(5,activation='sigmoid'))
model.add(Dense(4,activation='sigmoid'))
model.add(Dense(3,activation='sigmoid'))
model.add(Dense(2,activation='softmax'))
adam=optimizers.Adam(lr=.0006)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
snn=model.fit(x1,y1,validation_data=(x2,y2),batch_size=batch_size,epochs=epochs,shuffle=True)


plt.plot(snn.history['loss'],label='loss',marker="x", markersize="5", markeredgewidth="2")
plt.plot(snn.history['val_loss'],label='val_loss',marker="x", markersize="5", markeredgewidth="2")
sn.set(font_scale=.8)
plt.title("Loss")
plt.legend(loc='center right',fontsize='10')
plt.show()

plt.plot(snn.history['accuracy'],label='accuracy',marker="x", markersize="5", markeredgewidth="2")
plt.plot(snn.history['val_accuracy'],label='val_accuracy',marker="x", markersize="5", markeredgewidth="2")
sn.set(font_scale=.8)
plt.title("Accuracy")
plt.legend(loc='center right',fontsize='10')
plt.show()

#model.summary()


#Evaluation Test
evaluation = model.evaluate(x2,y2,batch_size=batch_size,verbose=1)
snn_pred = model.predict(x2, batch_size=batch_size) 
snn_predicted = np.argmax(snn_pred, axis=1)
snn_cm = confusion_matrix(y2, snn_predicted) 
snn_cmN= np.zeros((len(snn_cm),len(snn_cm)))
for i in range(len(snn_cm)):
	total=0
	for k in range(len(snn_cm)):
		total=total+snn_cm[i][k]
		total=total.astype(float)
	for j in range(len(snn_cm)):
		snn_cmN[i][j]=(snn_cm[i][j]/total)

snn_df_cm = pd.DataFrame(snn_cmN, range(2), range(2))
sn.set(font_scale=1.4)
sn.heatmap(snn_df_cm, annot=True,annot_kws={"size": 12})
plt.show()

print(x1.shape[1])
snn_report1 = classification_report(y2, snn_predicted,digits=4)
print(snn_report1)

