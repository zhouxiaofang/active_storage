import os.path
import torch.utils.data as data
import h5py
import glob
import time
import torch
from PIL import Image

global idx
idx = 0
class HDF5Dataset(data.Dataset):
    def __init__(self, h5_file, transform=None, target_transform=None):
        self.env = None
        self.path = h5_file
        self.transform = transform
        self.target_transform = target_transform
        # just simplify todo implemented
        self.length = 1252

    def __getitem__(self, index):
        if self.env is None:
            global idx
            idx = idx + 1
            print("number of h5 file: " + str(idx))
            now = time.time()
            with h5py.File(self.path, "r") as fh:
                self.images = fh["images"][()]
                self.labels = fh["labels"][()]
                self.heights = fh["heights"][()]
                self.widthes = fh["widthes"][()]
                self.length = len(self.labels)
                self.labels.resize((self.length,))
                print("eplased time: " + str(time.time() - now))
                self.env = fh
        img = self.images[index]
        img = img.reshape(self.heights[index], self.widthes[index], 3)
        lab = self.labels[index]
        if self.transform is not None:
            img = self.transform(Image.fromarray(img))
            #img = self.transform(img)
        if self.target_transform is not None:
            lab = self.target_transform(lab)
        return img, lab

    def __len__(self):
        return self.length

class HDF5Record(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.dbs = []
        h5_dirs = sorted(glob.glob(os.path.join(root, '*')))

        for h in h5_dirs:
            self.dbs.append(HDF5Dataset(
                h5_file=h,
                transform=transform,
                target_transform=target_transform))
        # todo    the length should be stored in another file
        self.indices = []
        count = 0
        for db in self.dbs:
            count += db.length
            self.indices.append(count)
        self.length = count

    def __getitem__(self, index):
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind
        db = self.dbs[target]
        index = index - sub
        img, lab = db[index]
        return img, lab

    def __len__(self):
        return self.length

