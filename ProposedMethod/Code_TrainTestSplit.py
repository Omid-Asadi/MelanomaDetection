#import packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array,array_to_img
from keras.utils import np_utils
import numpy as np
import pywt
from sklearn.metrics import average_precision_score,precision_score,recall_score
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import cv2
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import  Flatten, Dense, Activation,Convolution2D,MaxPooling2D,Dropout
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import statistics
def center_normalize(x):
    return (x-K.mean(x))/K.std(x)
def mymodel():
    model=Sequential()
#input layer
    model.add(Activation(activation=center_normalize, input_shape=(128, 128,1)))
# convolutional layer
    model.add(Convolution2D(32,5,5,border_mode='same',activation='relu',dim_ordering='tf'))
#pooling layer
    model.add(MaxPooling2D(pool_size=(2,2),dim_ordering='tf'))
# convolutional layer
    model.add(Convolution2D(64,3,3,border_mode='same',activation='relu',dim_ordering='tf'))
# pooling layer
    model.add(MaxPooling2D(pool_size=(2,2),dim_ordering='tf'))
    model.add(Flatten())
# Relu 
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
# Sigmoid Fully connected layer
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    return model
def w2d(imArray,imagePath):
    mode='db1'
    (ca, cd) = pywt.dwt(imArray,mode)
    cat = pywt.threshold(ca, np.std(ca)/250)
    cdt = pywt.threshold(cd, np.std(cd)/250)
    ts_rec = pywt.idwt(cat, cdt, mode)
    return np.array(cat)

data = []
labels = []
#Please customize the path
directory = "DWT_Yes"
for imagePath in os.listdir(directory):
    
    if imagePath.endswith(".jpg")  : 
        try:
                
                image = cv2.imread(directory + "//" + imagePath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #        image=w2d(image,imagePath)
        #        cv2.imwrite('DWT_Yes/'+imagePath,image)
                image = cv2.resize(image, (128,128),interpolation=cv2.INTER_AREA)
                image = img_to_array(image)
                data.append(image)
                labels.append(1)
        except:
                image = cv2.imread(directory + "//" + imagePath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #        image=w2d(image,imagePath)
        #        cv2.imwrite('DWT_Yes/'+imagePath,image)
                image = cv2.resize(image, (128,128),interpolation=cv2.INTER_AREA)
                image = img_to_array(image)
                data.append(image)
                labels.append(1)
                
#Please customize the path
directory = "DWT_No"
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

for imagePath in os.listdir(directory):
    if imagePath.endswith(".jpg"): 
        try:
                image = cv2.imread(directory + "//" + imagePath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #        image=w2d(image,imagePath)
        #        blur = cv2.GaussianBlur(image,(5,5),0)
        #        ret3,image = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
        #        cv2.imwrite('DWT_No/'+imagePath,image)
        
                image = cv2.resize(image, (128,128),interpolation=cv2.INTER_AREA)
                image = img_to_array(image)
                data.append(image)
                labels.append(0)
        except:
            
                image = cv2.imread(directory + "//" + imagePath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #        image=w2d(image,imagePath)
        #        blur = cv2.GaussianBlur(image,(5,5),0)
        #        ret3,image = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
        #        cv2.imwrite('DWT_No/'+imagePath,image)
        
                image = cv2.resize(image, (128,128),interpolation=cv2.INTER_AREA)
                image = img_to_array(image)
                data.append(image)
                labels.append(0)
                
data = np.array(data, dtype="float") 
labels = np.array(labels)
print(labels,len(labels))
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals
skf=StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
# initialize the model
Results=[]
reallabels=[]
predictionslabel=[]
AUCC=[]
PR=[]
REC=[]
print()
trainX, testX, trainY, testY = train_test_split(data, labels,stratify=labels, test_size=0.20, random_state=1)

generator = tf.keras.preprocessing.image.ImageDataGenerator(
horizontal_flip = True,
vertical_flip = True,
rotation_range = 30)

le = LabelEncoder().fit(labels)
trainY = np_utils.to_categorical(le.transform(trainY), 2)
#    le = LabelEncoder().fit(testY)
testY = np_utils.to_categorical(le.transform(testY), 2)
print("[INFO] compiling model...")
model=mymodel()
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])
H = model.fit(train_datagen.flow(trainX, trainY, batch_size = 32), validation_data=(testX, testY), steps_per_epoch=len(trainX) // 32,epochs=5, verbose=1)
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=20)
precision=precision_score(testY.argmax(axis=1), predictions.argmax(axis=1), average='weighted')
average_precision = average_precision_score(testY, predictions)
recall = recall_score(testY.argmax(axis=1), predictions.argmax(axis=1),average='weighted')
AUCC.append(average_precision)
REC.append(recall)
PR.append(precision)
print('AUPR',average_precision,'presi',precision,'recall',recall)
#    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))
model.save('mymodel.HDF5')

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
 fpr[i], tpr[i], _ = metrics.roc_curve(testY[:, i], predictions[:, i])
 roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = metrics.roc_curve(testY.ravel(), predictions.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])   
Results.append(roc_auc)
Results.append(H.history)


print(statistics.mean(AUCC),statistics.mean(PR),statistics.mean(REC))
