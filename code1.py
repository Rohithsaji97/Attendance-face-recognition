from keras.applications.vgg16 import VGG16
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import Dense,Flatten
from tensorflow.keras.optimizers import RMSprop


os.chdir("R:/Face/model")
train= ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)

train_dataset=train.flow_from_directory("R:/Face/model/train/",
                                        target_size=(200,200),
                                        batch_size=60,
                                        class_mode='categorical')

validation_dataset=validation.flow_from_directory("R:/Face/model/validation/",
                                        target_size=(200,200),
                                        batch_size=42,
                                        class_mode='categorical')

conv_base = VGG16(
    weights='imagenet',
    include_top = False,
    input_shape=(200,200,3)
)

conv_base.trainable = False

conv_base.trainable = True

set_trainable = False

for layer in conv_base.layers:
  if layer.name == 'block5_conv1':
    set_trainable = True
  if set_trainable:
    layer.trainable = True
  else:
    layer.trainable = False



model = Sequential()

model.add(conv_base)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(3,activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer=RMSprop(learning_rate=0.001),metrics=['accuracy'])

model_fit=model.fit(train_dataset,
                    steps_per_epoch=20,
                    epochs=10,
                    validation_data=validation_dataset,
                    validation_steps=100)


model.save('FR1.h5')

