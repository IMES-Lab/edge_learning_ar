import time
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import os

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
output_size = 10

# dropout (keep_prob) rate (should be 1 for testing)
keep_prob = 0.3

MODEL_SAVE_FOLDER_PATH = './MNIST_data/'

# dataset setting
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
    os.mkdir(MODEL_SAVE_FOLDER_PATH)
model_path = MODEL_SAVE_FOLDER_PATH + 'mnist-' + '{epoch:02d}-{val_loss:.4f}.hdf5'
cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)
(X_train, Y_train), (X_validation, Y_validation) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_validation = X_validation.reshape(X_validation.shape[0], 28, 28, 1).astype('float32') / 255

Y_train = np_utils.to_categorical(Y_train, 10)
Y_validation = np_utils.to_categorical(Y_validation, 10)

# control parameter
filter_size = 3
moveX = 1
moveY = 1
pooling_dist = 2
depth_size = 32
depth_increase = 2
layer_size = 4

start_time = time.time()

model = Sequential()

# First Layer (input image = (28, 28, 1), output layer = (28, 28, 32))
model.add(Conv2D(depth_size, kernel_size=(filter_size, filter_size), strides=(moveX, moveY), input_shape=(28, 28, 1),
                 activation='relu', padding='SAME'))
model.add(MaxPooling2D(pool_size=(pooling_dist, pooling_dist)))
model.add(Dropout(keep_prob))

# Hidden Layer
for index in range(1, layer_size):
    layer_depth = (int)(depth_size * pow(depth_increase, index))
    model.add(Conv2D(layer_depth, kernel_size=(filter_size, filter_size),
                     strides=(moveX, moveY), activation='relu', padding='SAME'))
    model.add(MaxPooling2D(pool_size=(pooling_dist, pooling_dist)))
    model.add(Dropout(keep_prob))

# FC Layer
model.add(Flatten())
model.add(Dense(depth_size * pow(depth_increase, layer_size), activation='relu'))
model.add(Dropout(keep_prob))
model.add(Dense(output_size, activation='softmax'))

# training setting
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Training start')

history = model.fit(X_train, Y_train,
                    validation_data=(X_validation, Y_validation),
                    epochs=training_epochs, batch_size=batch_size, verbose=0,
                    callbacks=[cb_checkpoint, cb_early_stopping])

print('\nAccuracy: {:.4f}'.format(model.evaluate(X_validation, Y_validation)[1]))

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

# check learning time
print('Training end')
end_time = time.time()
print("Learning time: %d minutes %0.2f seconds\n" % ((int)((end_time - start_time)/60), (end_time - start_time) % 60))
