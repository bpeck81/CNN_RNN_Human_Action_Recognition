import torch
from torch.utils.data import Dataset
import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from torchvision import models, transforms
import json

class VideoLoader(Dataset):

    def __init__(self, dataset_name, transforms, root_dir, seq_len):
        self.labels = []
        self.data_file_names = []
        self.data_file_labels = []
        self.root_dir = root_dir
        self.transforms = transforms
        self.seq_len = seq_len
        datset_dir = os.path.join(self.root_dir, dataset_name)
        comp_exts = ['.rar', '.zip', '.7z']
        video_exts = ['.avi', '.mp4']
        for dir_name in sorted(os.listdir(datset_dir)):
            if True in [ext in dir_name for ext in comp_exts]: continue
            self.labels.append(dir_name)
            class_dir = os.path.join(datset_dir, dir_name)
            for file_name in os.listdir(class_dir):
                if True not in [ext in file_name for ext in video_exts]: continue
                file_name = os.path.join(class_dir, file_name)
                self.data_file_names.append(file_name)
                self.data_file_labels.append(dir_name)


    def __len__(self):
        return len(self.data_file_names)

    def __getitem__(self, idx):
        try:
            file_name = self.data_file_names[idx]
            cap = imageio.get_reader(file_name, 'ffmpeg')
            frames = []
            for num, frame in enumerate(cap):
                if self.transforms is not None:
                    frame = self.transforms(frame)
                frames.append(frame)
            if len(frames) < self.seq_len: return None
            if self.seq_len != None:
                frames = frames[-self.seq_len:]
            frames = torch.squeeze(torch.stack(frames))
            label = self.labels.index(self.data_file_labels[idx])
            label = torch.tensor([label]).squeeze()
            return frames, label
        except Exception as e:
            print(self.data_file_names[idx])
            print(e)
            return None
