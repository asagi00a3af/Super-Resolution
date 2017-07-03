'''
動画データセットの作成スクリプト
動画データは事前に非圧縮の画像データに連番で切り分ける,
画像データの形式はtiffとする
'''
import numpy as np
import h5py
import cv2
from os import path
import glob

#データセットパラメータ
#三次元データ (Nf, H, W)
W = 36
H = 36
Nf = 5
#パス情報
images_path = '/media/shimo/HDD_storage/DataSet/SCENE_1/2K_images/*.tiff'
#画像連番リスト
image_names = sorted(glob.glob(images_path))
#ソース画像のサイズ取得
src_h, src_w, ch = cv2.imread(image_names[0]).shape
del ch
#全テー多数
num_data = (src_h // H) * (src_w // W) * (len(image_names) // Nf)

#hdf5関連設定
idx = 0
f = h5py.File('test.hdf5','w')
f.create_dataset('dat', (num_data, H, W, Nf), chunks=True, dtype=np.float32)
f.create_dataset('lab', (num_data, H, W),chunks=True, dtype=np.float32)

for seq in range(len(image_names) // Nf):
    '''
    seq: sequence
    画像連番リストをフレーム数で分割し,
    そのフレーム数のデータセットごと画像を切り出す
    '''
    print(seq, "/", len(image_names) // Nf)
    image_data = np.zeros((src_h, src_w, Nf))
    #Nf枚の画像読み込み
    for i, name in enumerate(image_names[seq * Nf : (seq + 1) * Nf]):
        image_data[:,:,i] = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2YCR_CB)[:,:,0]
    #読み込んだ画像をラベルとデータセットの形状に変換
    image_label = image_data[:,:,Nf//2+1].astype(np.float32) / 255
    image_buf = cv2.resize(image_data, (src_w//3, src_h//3), interpolation=cv2.INTER_CUBIC)
    image_data = cv2.resize(image_buf, (src_w, src_h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    for y in range(src_h//H):
        for x in range(src_w//W):
            f['dat'][idx, :, :, :] = image_data[y*H:(y+1)*H, x*W:(x+1)*W, :]
            f['lab'][idx, :, :] = image_label[y*H:(y+1)*H, x*W:(x+1)*W]
            idx += 1
    f.flush()
f.close()
