
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
def ZFNET():
   
    N_CATEGORY=2
    model_input = Input(shape =(128,128,1))
    
    # First convolutional Layer (96x7x7)
    z = Conv2D(filters = 96, kernel_size = (7,7), strides = (2,2), activation = "relu")(model_input)
    z = ZeroPadding2D(padding = (1,1))(z)
    z = MaxPooling2D(pool_size = (3,3), strides=(2,2))(z)
    z = BatchNormalization()(z)
    
    # Second convolutional Layer (256x5x5)
    z = Convolution2D(filters = 256, kernel_size = (5,5), 
                      strides = (2,2), activation = "relu")(z)
    z = ZeroPadding2D(padding = (1,1))(z)
    z = MaxPooling2D(pool_size = (3,3), strides=(2,2))(z)
    z = BatchNormalization()(z)
    
    # Rest 3 convolutional layers
    z = ZeroPadding2D(padding = (1,1))(z)
    z = Convolution2D(filters = 384, kernel_size = (3,3), 
                      strides = (1,1), activation = "relu")(z)
    
    z = ZeroPadding2D(padding = (1,1))(z)
    z = Convolution2D(filters = 384, kernel_size = (3,3), 
                      strides = (1,1), activation = "relu")(z)
    
    z = ZeroPadding2D(padding = (1,1))(z)
    z = Convolution2D(filters = 256, kernel_size = (3,3), 
                      strides = (1,1), activation = "relu")(z)
    
    z = MaxPooling2D(pool_size = (3,3), strides=(2,2))(z)
    z = Flatten()(z)
    
    z = Dense(4096, activation="relu")(z)
    z = Dropout(DROPOUT)(z)
    
    z = Dense(4096, activation="relu")(z)
    z = Dropout(DROPOUT)(z)
    

    model_output = Dense(2, activation='sigmoid')(z)
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
skf=StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
# initialize the model
Results=[]
reallabels=[]
predictionslabel=[]
AUCC=[]
PR=[]
REC=[]
i=0
for train_index, test_index in skf.split(data, labels):
    i=i+1
    trainX, testX = data[train_index], data[test_index]
    trainY, testY = labels[train_index], labels[test_index]
    generator = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip = True,
    vertical_flip = True,
    rotation_range = 30)
    le = LabelEncoder().fit(labels)
    trainY = np_utils.to_categorical(le.transform(trainY), 2)
#    le = LabelEncoder().fit(testY)
    testY = np_utils.to_categorical(le.transform(testY), 2)
    print("[INFO] compiling model...")
    model=ZFNET()
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
    model.save('ZFNET'+str(i)+'.HDF5')
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
