import pandas as pd
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

audio_dataset_path='Sound Dataset/'
num_labels = 10


def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features

extracted_features=[]
'''
for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])
'''
for folder in os.listdir(audio_dataset_path):
    print(folder)
    for file_name in os.listdir(audio_dataset_path+'/'+folder+'/'):
        data = features_extractor(audio_dataset_path+'/'+folder+'/'+file_name)
        extracted_features.append([data,folder])

extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])

X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())

target = y
inputs = X
print(len(X))
print(len(y))


from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))
print(y)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

num_epochs = 200
num_batch_size = 2

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)

test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])

#Saving the model........................
import h5py
model.save('sound_model.h5')



filename="Sound Dataset/Alectoris Chukar/XC133064 - Chukar Partridge - Alectoris chukar.wav"
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

print(mfccs_scaled_features)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
print(mfccs_scaled_features)
print(mfccs_scaled_features.shape)
predicted_label=model.predict(mfccs_scaled_features)
print(predicted_label)
classes_x=np.argmax(predicted_label,axis=1)
prediction_class = labelencoder.inverse_transform(classes_x)
print(prediction_class)




#Step5: Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = inputs
x_train = sc.fit_transform(X)

# Import label encoder
from sklearn.preprocessing import LabelEncoder
 
# label_encoder object knows how to understand word labels.
label_encoder = LabelEncoder()
y = target
y = label_encoder.fit_transform(y)


#Step6: Fitting the KNN classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y)

#Step7: Performance
from sklearn.metrics import confusion_matrix,accuracy_score
y_pred = classifier.predict(x_train)
cm = confusion_matrix(y,y_pred)
print(cm)
print("Accuracy of KNN:",accuracy_score(y, y_pred))


#Fitting the Random Forest classifier----------------------
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='entropy',n_estimators=10)
classifier.fit(x_train,y)

#Step7: Performance
from sklearn.metrics import confusion_matrix,accuracy_score
y_pred = classifier.predict(x_train)
cm = confusion_matrix(y,y_pred)
print(cm)
print("Accuracy of RFC:",accuracy_score(y, y_pred))

#Fitting the Decision Tree classifier----------------------
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(x_train,y)

#Step7: Performance
from sklearn.metrics import confusion_matrix,accuracy_score
y_pred = classifier.predict(x_train)
cm = confusion_matrix(y,y_pred)
print(cm)
print("Accuracy of DTC:",accuracy_score(y, y_pred))

#Fitting the model
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear',random_state=0)
classifier.fit(x_train,y)

#Step7: Performance
from sklearn.metrics import confusion_matrix,accuracy_score
y_pred = classifier.predict(x_train)
cm = confusion_matrix(y,y_pred)
print(cm)
print("Accuracy of SVM:",accuracy_score(y, y_pred))