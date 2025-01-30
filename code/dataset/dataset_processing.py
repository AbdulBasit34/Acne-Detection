import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np

class DatasetProcessing(Dataset):
    def __init__(self, data_path, img_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        print(f"Initializing DatasetProcessing with data path: {data_path} and image file: {img_filename}")
        # reading img file from file
        img_filepath = img_filename
        fp = open(img_filepath, 'r')

        self.img_filename = []
        self.labels = []
        self.lesions = []
        for line in fp.readlines():
            filename, label, lesion = line.split()
            self.img_filename.append(filename)
            self.labels.append(int(label))
            self.lesions.append(int(lesion))
        fp.close()
        print(f"Loaded {len(self.img_filename)} images from {img_filename}")

        self.img_filename = np.array(self.img_filename)
        self.labels = np.array(self.labels)
        self.lesions = np.array(self.lesions)

        if 'NNEW_trainval' in img_filename:
            ratio = 1.0
            import random
            random.seed(42)
            indexes = []
            for i in range(4):
                index = random.sample(list(np.where(self.labels == i)[0]), int(len(np.where(self.labels == i)[0]) * ratio))
                indexes.extend(index)
            self.img_filename = self.img_filename[indexes]
            self.labels = self.labels[indexes]
            self.lesions = self.lesions[indexes]
            print(f"After balancing, dataset size is {len(self.img_filename)}")

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.img_filename[index])
        #print(f"Loading image: {img_path}")
        img = Image.open(img_path)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        name = self.img_filename[index]
        label = torch.from_numpy(np.array(self.labels[index]))
        lesion = torch.from_numpy(np.array(self.lesions[index]))
        return img, label, lesion

    def __len__(self):
       # print(f"Dataset length requested: {len(self.img_filename)}")
        return len(self.img_filename)

