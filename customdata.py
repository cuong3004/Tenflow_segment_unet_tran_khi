import cv2
import numpy as np 
from tensorflow.keras.utils import Sequence
import pydicom
from rle2mask import rle2mask

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self,
                 all_filenames,
                 batch_size,
                 input_dim,
                 n_channels,  
                 transform,
                 shuffle=True):
        
        self.all_filenames = all_filenames
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.n_channels = n_channels
        self.transform = transform
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        '''
        return:
          Trả về số lượng batch/1 epoch
        '''
        return int(np.floor(len(self.all_filenames) / self.batch_size))

    def __getitem__(self, index):
        '''
        params:
          index: index của batch
        return:
          X, y cho batch thứ index
        '''
        # Lấy ra indexes của batch thứ index
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # List all_filenames trong một batch
        indexs = [k for k in indexes]
        # Khởi tạo data
        X, Y = self.__data_generation(indexs)
        return X, Y

    def on_epoch_end(self):
        '''
        Shuffle dữ liệu khi epochs end hoặc start.
        '''
        self.indexes = np.arange(len(self.all_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexs):
        #print("ok")
        '''
        params:
          all_filenames_temp: list các filenames trong 1 batch
        return:
          Trả về giá trị cho một batch.
        '''
        X = np.empty((self.batch_size, *self.input_dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.input_dim, self.n_channels))
        # Khởi tạo dữ liệu
        for i, index in enumerate(indexs):
            name = self.all_filenames.values[index][-1]
            img = pydicom.read_file(name).pixel_array
            pixel = self.all_filenames.values[index][1]
            if pixel != ' -1':
                label = rle2mask(pixel, 1024, 1024)
                label = np.rot90(label, 3) #rotating three times 90 to the right place
                label = np.flip(label, axis=1)
            else: 
                label = np.zeros((512,512,1))
            if self.transform is not None:
                img, label = self.transform(img, label)
            X[i,] = img
            Y[i] = label
            del img, label, name, pixel
        return X, Y