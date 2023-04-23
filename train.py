import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
dataset=tf.keras.utils.image_dataset_from_directory("Training/",class_names=["female","male"],image_size=(83,108),batch_size=64)
dataset=dataset.map(lambda x,y:((x/255),tf.one_hot(y,2)))
val=tf.keras.utils.image_dataset_from_directory("Validation/",class_names=["female","male"],image_size=(83,108),batch_size=64)
val=val.map(lambda x,y:(x/255,tf.one_hot(y,2)))
#model
inputs=Input(shape=(83,108,3))
conv=Conv2D(3,4,activation="relu")(inputs)
pool=MaxPooling2D()(conv)
norm=BatchNormalization()(pool)
conv2=Conv2D(3,4,activation="relu")(norm)
pool2=MaxPooling2D()(conv2)
norm2=BatchNormalization()(pool2)
att=AdditiveAttention()([norm2,norm2])
flat=Flatten()(att)
out=Dense(2,activation="sigmoid")(flat)
model=tf.keras.Model(inputs=inputs,outputs=out)
opt=tf.keras.optimizers.Adam()
loss=tf.keras.losses.CategoricalCrossentropy()
metric=[tf.keras.metrics.Recall(),tf.keras.metrics.CategoricalAccuracy()]
model.compile(opt,loss,metric)
model.fit(dataset,batch_size=64,epochs=100,validation_data=val)

model.save("saved/model")
