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
            fh = h5py.File(self.path, "r")
            img = fh["images"][index, :, :, :]
            lab = fh["labels"][index]
            self.length = 1252
            print("eplased time: " + str(time.time() - now))
            self.env = fh
        else:
#            now = time.time()
            img = self.env["images"][index, :, :, :]
            lab = self.env["labels"][index]
#            print("22222222222 eplased time: " + str(time.time() - now))
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

