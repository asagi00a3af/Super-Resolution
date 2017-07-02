import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

batch_size = 64
epochs = 10
# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = x_train
y_test = x_test

input_shape = x_train.shape
print(x_train.shape[1:])
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=9, strides=1,padding='same',
                input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=5, strides=1,padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=3, kernel_size=5, strides=1,padding='same'))

opt = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.00001, clipnorm=2.)
model.compile(loss='mean_squared_error', optimizer=opt)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=None,shuffle=True)
