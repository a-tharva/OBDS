#Classification Using Audio....................................................
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


for folder in os.listdir(audio_dataset_path):
    # print(folder)
    for file_name in os.listdir(audio_dataset_path+'/'+folder+'/'):
        data = features_extractor(audio_dataset_path+'/'+folder+'/'+file_name)
        extracted_features.append([data,folder])

extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])

X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())

target = y
inputs = X
# print(len(X))
# print(len(y))

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

#Fitting the model
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear',random_state=0)
classifier.fit(x_train,y)

#Step7: Performance
from sklearn.metrics import confusion_matrix,accuracy_score
y_pred = classifier.predict(x_train)
cm = confusion_matrix(y,y_pred)
# print(cm)
# print("Accuracy of SVM:",accuracy_score(y, y_pred))


BIRDS = [
    "Alectoris Chukar",
    "Ashy minivet",
    "Asian Brown Flycatcher",
    "Black Headed Greenfinch",
    "Black Throtted Babler",
    "Common Kingfisher",
    "Japanese Green Woodpecker",
    "Scarlet Rumped Trogon",
    "Soundscape",
    "White Crowned Forktail",
]



def soundPredict():
    filename="upload/test.wav"

    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

    # print(mfccs_scaled_features)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    # print(mfccs_scaled_features)
    print(mfccs_scaled_features.shape)
    mfccs_scaled_features = sc.transform(mfccs_scaled_features)
    predicted_label=classifier.predict(mfccs_scaled_features)
    print(predicted_label)

    predicted_class = BIRDS[predicted_label[0]]

    return(predicted_class)
