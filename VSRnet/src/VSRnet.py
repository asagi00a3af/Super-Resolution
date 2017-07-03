import numpy as np
import h5py
import keras
from keras.models import Model
from keras.layers import Conv2D, Input, concatenate
from keras.callbacks import ModelCheckpoint

#データパス
hdf5_data_path = '/media/shimo/HDD_storage/DataSet/SCENE_1/SCENE1_2K.hdf5'

#パラメータ
batch_size = 128
epoch = 100

if __name__ == '__main__':
    #データ読み込み
    with h5py.File(hdf5_data_path, 'r') as f:
        x_data = f['dat'].value
        y_data = f['lab'].value

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_train.shape[0], 'test samples')

    input_shape = x_train.shape[1:]

    #モデルデザイン
    input_seq = (shape=input_shape)
