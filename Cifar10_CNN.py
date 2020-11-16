'''
Accuracy:78.1%
Loss:1.62
'''
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import cifar10   #111
from matplotlib import pyplot
from keras.utils import np_utils


print('Python version : ', sys.version)
print('Keras version : ', keras.__version__)

img_rows = 32   #111
img_cols = 32
num_classes=10


(x_train, y_train), (x_test, y_test) =cifar10.load_data()

y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



epochs = 250
filename='checkpoint.h5'.format(epochs)
early_stopping=EarlyStopping(monitor='val_loss',mode='min',patience=15,verbose=1)                           #얼리스타핑
checkpoint=ModelCheckpoint(filename,monitor='val_loss',verbose=1,save_best_only=True,mode='auto')           #체크포인트



model = Sequential()
#1
model.add(Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', input_shape=x_train.shape[1:]))
model.add(Conv2D(64, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(256, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#2
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

opt=keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=epochs, verbose=1, validation_data=(x_test, y_test),callbacks=[checkpoint,early_stopping]) #학습

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:',  score[0])
print('Test accuracy:', score[1])
