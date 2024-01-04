import os
import torch
import torchvision.transforms as transforms
import scipy.io as sio
import numpy as np
import cv2


class Dataset(torch.utils.data.Dataset):

    def __init__(self, mat_path):
        # 1.获取文件路径
        self.data_path = mat_path
        self.images = self.read_file()

    def __getitem__(self, index):
        data = sio.loadmat(self.images[index])
        pan = data['Z']
        lrhs = data['Y']
        hs = data['label']
        lrhs = np.transpose(lrhs, (2, 0, 1)).astype(np.float32)
        lrhs = torch.from_numpy(lrhs)
        pan = np.transpose(pan, (2, 0, 1)).astype(np.float32)
        pan = torch.from_numpy(pan)
        hs = np.transpose(hs, (2, 0, 1)).astype(np.float32)
        hs = torch.from_numpy(hs)
        return {'Z': pan, 'Y': lrhs, 'X': hs}

    def __len__(self):
        return len(self.images)

    def read_file(self):
        # 1.读取mat文件
        path_list = []
        for ph in os.listdir(self.data_path):
            path = self.data_path + ph
            path_list.append(path)
        return path_list
