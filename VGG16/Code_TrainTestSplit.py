
#import packages
import keras
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score,precision_score,recall_score
from keras.layers import Conv2D, Convolution2D, MaxPooling2D
from keras.layers import Input
from keras.layers.convolutional import ZeroPadding2D
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
from keras.optimizers import Adam,SGD # I believe this is better optimizer for our case
from keras.preprocessing.image import ImageDataGenerator # to augmenting our images for increasing accuracy
from keras.utils.vis_utils import plot_model
import scipy
from keras.models import Model
import cv2
from sklearn.model_selection import train_test_split # to split our train data into train and validation sets
import numpy as np
import pandas  as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array,array_to_img
from keras.utils import np_utils
import numpy as np
import pywt
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
def VGG16():
        # Block 1
    model_input = Input(shape = (128, 128,1))
    x = Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(model_input)
    x = Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)

    model_output = Dense(2, activation='sigmoid', name='predictions')(x)
    model = Model(model_input, [model_output])
    return model
def center_normalize(x):
    return (x-K.mean(x))/K.std(x)

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
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals
# initialize the model
Results=[]
reallabels=[]
predictionslabel=[]
AUCC=[]
PR=[]
REC=[]
i=0
trainX, testX, trainY, testY= train_test_split(data, labels,stratify=labels, test_size=0.20, random_state=1)
generator = tf.keras.preprocessing.image.ImageDataGenerator(
horizontal_flip = True,
vertical_flip = True,
rotation_range = 30)
le = LabelEncoder().fit(labels)
trainY = np_utils.to_categorical(le.transform(trainY), 2)
#    le = LabelEncoder().fit(testY)
testY = np_utils.to_categorical(le.transform(testY), 2)
print("[INFO] compiling model...")
model=VGG16()
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
model.save('VGG16.HDF5')
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
