
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets
import copy
import numpy as np
import random
import time
import sys


def make_rand(net_list):
  lis=list()
  re_seed=random.randint(1,4) 
  for i in range(re_seed):
    seed=random.randint(1,4) 
    if seed==1:
     im_output= layers.Conv2D(filters=64, kernel_size=[ks,ks], padding='same', activation=actF)(output)
    elif seed==2:
      im_output= layers.Dropout(rate=drop_out)(output)
    elif seed==3:
     im_output= layers.MaxPooling2D(pool_size=[ks, ks], padding='same', strides=1)(output)
    elif seed==4:
     im_output = layers.Activation(actF)(output)
    lis.append(im_output)
  return lis
start_time = time.time()
start_clock = time.clock()
lr = 1.0080999752044677
initW = 'None'
opt = keras.optimizers.Adagrad(learning_rate=lr)
actF = 'tanh'
ks = 5
depth = 2
fc_layer = 1
drop_out = 0.42299808221717194
byp = 4

img_rows = 28
img_cols = 28
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
(x_train, x_test) = x_train[:30000], x_test[:1000]
(y_train, y_test) = y_train[:30000], y_test[:1000]
input_shape = (img_rows, img_cols, 1)
batch_size = 128
num_classes = 10
epochs =10

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

inputs = keras.Input(shape = input_shape, name = 'input')
output = layers.Conv2D(filters = 64, kernel_size = [ks, ks], padding = 'same', activation = actF)(inputs)

net_list=list()
add_num=0
for _ in range(depth):
  a=make_rand(net_list)
  net_list.extend(a)
  if len(a)==1:r_num=0 
  else:r_num=random.randint(0,len(a)-1)                 
  output=a[r_num]   
  
  short_cut_dec=random.randint(1,5)     
  if (short_cut_dec<=byp) and len(net_list)>1:
    add_num=add_num+1
    for _ in range( random.randint(0,len(net_list)-1) ):
      a_layer_num=random.randint(0,len(net_list)-1)
      if a_layer_num!=r_num:
       c=layers.Add()([net_list[a_layer_num],output])
       output=c
       net_list.append(c)
output = layers.GlobalAveragePooling2D()(output)
output = layers.Dense(1000, activation = actF, name='fc0')(output)
dropout = layers.Dropout(rate=drop_out)(output)
output = layers.Dense(10, activation = 'softmax', name='output')(dropout)

model = keras.Model(inputs = inputs, outputs = output)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
end_time = time.time()
end_clock = time.clock()
train_time = end_time - start_time
train_clock = end_clock - start_clock
print('Time to train (time) = ', train_time)
print('Time to train (clock) = ', train_clock)
print("Accuracy=", score[1], "genetic")
model.save('./saved/model_9.h5')
