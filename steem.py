from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard,ModelCheckpoint
batch_size = 32

#augmenting our images during training to offset small dataset
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

#we do not augement the images during validation
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        'data/train',  # source dir
        target_size=(100, 100),  # resizing images to 150x150(square)
        batch_size=batch_size,
        class_mode='binary')  # using binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/validate',
        target_size=(100, 100),
        batch_size=batch_size,
        class_mode='binary')


#constructing our model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 100, 100)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#max pooling to downsample

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # 3d feature maps --> 1d feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])




checkpoint=ModelCheckpoint(filepath='model.h5',monitor='val_acc',mode='auto',save_best_only=True)
tensor = TensorBoard(write_grads=True,write_graph=True,write_images=True,log_dir='logs')

model.fit_generator(
        train_generator,
        steps_per_epoch=2930 // batch_size,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=978 // batch_size,
        callbacks=[tensor,checkpoint])


from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

img = load_img('10_100.jpg')
img = img_to_array(img)
img = img.reshape((1,) + img.shape)
print img.shape
model = load_model('model.h5')
pred = model.predict(img)
print pred