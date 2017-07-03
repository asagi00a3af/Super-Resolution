from os import path
import argparse
import glob
import numpy as np
import h5py
import cv2
import chainer
import chainer.functions as F
import chainer.links as L

#ネットワークに関するパラメータ
test_batch_size = 2
train_batch_size = 64
grad_clip = 0.1

class SeqDataset(chainer.dataset.DatasetMixin):
    '''
    シーケンスデータセットクラス
    シーケンスデータ:H*W*Nf(縦*幅*フレーム数)の3次元データ
    '''
    def __init__(self, data_path):
        with h5py.File(datapath, 'r') as f:
            self.x_data = f['dat'].value
            self.y_data = f['lab'].value
        self.length = x_data.shape[0]

    def __len__(self):
        return self.length

    def get_example(self, i):
        return x_data[i], y_data[i]

class VSRnet(chainer.chain):
    '''
    Video Super Resolution network
    '''
    def __init__(self):
        super(VSRnet, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(1, 32)
