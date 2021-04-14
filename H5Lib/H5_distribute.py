import os.path
import torch.utils.data as data
import h5py
import glob
import time
import torch
from PIL import Image
import numpy as np
import gc
from base.common import print_memory_info


class HDF5Record(data.Dataset):
    def __init__(self, root, gap, batch_size, worker_count, h5_package_size, world_size=1, transform=None, target_transform=None):
        self.batch_size = batch_size
        self.worker_count = worker_count
        self.h5_package_size = h5_package_size
        self.world_size = world_size
        self.dbs = []
        h5_dirs = sorted(glob.glob(os.path.join(root, '*.h5')))
        len_array = np.load(os.path.join(root, 'length.npy'))
        self.biggest_index = sum(len_array) - 1

        index = 0
        for h in h5_dirs:
            self.dbs.append(HDF5Dataset(
                h5_file=h,
                gap=gap,
                transform=transform,
                target_transform=target_transform, length=len_array[index]))
            index = index + 1
        self.indices = []
        count = 0
        for db in self.dbs:
            count += db.length
            self.indices.append(count)
            print("h5 db length: {0}".format(db.length))
        
        if self.world_size > 1:
            # distribute case
            self.no_map_index = self.biggest_index // (batch_size * h5_package_size * worker_count * world_size) * (batch_size * h5_package_size * worker_count * world_size)
            print("h5 dbs lengths: {0}, biggest index: {1}, no map index: {2}".format(len(self.dbs), self.biggest_index, self.no_map_index))
        elif self.worker_count > 1:
            # no distribute case, but multiple processes case
            self.no_map_index = self.biggest_index // (batch_size * h5_package_size * worker_count) * (batch_size * h5_package_size * worker_count)
            print("h5 dbs lengths: {0}, biggest index: {1}, no map index: {2}".format(len(self.dbs), self.biggest_index, self.no_map_index))
        else:
            print("h5 dbs lengths: {0}, biggest index: {1}".format(len(self.dbs), self.biggest_index))

        self.length = count

    def __getitem__(self, index):
        print('original index: {0}'.format(index))
        if self.world_size > 1:
            index = self._re_mapping_index_distribute(index)
        elif self.worker_count > 1:
            index = self._re_mapping_index(index)
        print('mapped new index: {0}'.format(index))

        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind

        print('Access db target: {0}'.format(target))
        db = self.dbs[target]
        print('Access db file: {0}'.format(db.file))
        index = index - sub
        img, lab = db[index]
        return img, lab
    
    def _re_mapping_index(self, index):
        ''' 
        Used for re-mapping input index
        Premise:
        1. use h5p package
        2. use more than 1 subProcess.

        e.g. 
            batch size: 10 
            worker count: 3
            h5 package size: 3
            so, 
                                worker0                           worker1                          worker2
            batch_idx     index       h5 index               index       h5 index              index      h5 index      batch_idx / h5_package_size      batch_idx % h5_package_size
               0           0 - 9   |   0 - 9                10 - 19   |   30 - 39            20 - 29   |   60 - 69                 0                              0
               1          30 - 39  |  10 - 19               40 - 49   |   40 - 49            50 - 59   |   70 - 79                 0                              1
               2          60 - 69  |  20 - 29               70 - 79   |   50 - 59            80 - 89   |   80 - 89                 0                              2

               3          90 - 99  |  90 - 99              100 - 109  |  120 - 129          110 - 119  |  150 - 159                1                              0
               4         120 - 129 | 100 - 109             130 - 139  |  130 - 139          140 - 149  |  160 - 169                1                              1
               5         150 - 159 | 110 - 119             160 - 169  |  140 - 149          170 - 179  |  170 - 179                1                              2

               6         180 - 189 | 180 - 189             190 - 199  |  210 - 219          200 - 209  |  240 - 249                2                              0
                                                             ...

            mapping the input index to the new h5 index.
            step 1:
            h5_size = h5_package_size * batch_size
            remainder = index % batch_size

            step 2:
            worker_id = (index / batch_size) % worker_count
            batch_idx = (index / batch_size - worker_id) / worker_count

            step 3:
            h5_index = (batch_idx / h5_package_size) * h5_size * worker_count + worker_id * h5_size + (batch_idx % h5_package_size) * batch_size + remainder

        '''
        h5_size = self.h5_package_size * self.batch_size
        if index >= self.no_map_index:
            return index

        current_remainder = index % self.batch_size
        worker_id = (int(index / self.batch_size)) % self.worker_count
        batch_idx = int((int(index / self.batch_size) - worker_id) / self.worker_count)
        h5_index = int(batch_idx / self.h5_package_size) * h5_size * self.worker_count + worker_id * h5_size + (batch_idx % self.h5_package_size) * self.batch_size + current_remainder
        return h5_index

    def _re_mapping_index_distribute(self, index):
        ''' 
        Used for re-mapping input index for distributed training model.
        Premise:
        1. at least 2 computing nodes
        2. use h5p package
        '''
        if index >= self.no_map_index:
            return index
        
        gap_size = self.world_size * self.batch_size
        node_id = index % self.world_size
        h5_size = self.batch_size * self.h5_package_size
        # whole_size means the count of pictures it needed if each subProcess per node could get one h5 package data
        whole_size = h5_size * self.worker_count * self.world_size
        worker_id = index // gap_size % self.world_size
        current_remainder = (index - node_id) % gap_size / self.world_size

        A = index // whole_size
        B = index % whole_size // (self.world_size * self.worker_count * self.batch_size)
        C = index // (self.world_size * self.worker_count * self.batch_size)

        h5_index = A * whole_size + node_id * h5_size + worker_id * h5_size * self.world_size + B * self.batch_size + current_remainder
        return h5_index

    def __len__(self):
        return self.length


class HDF5Dataset(data.Dataset):
    def __init__(self, h5_file, gap = 0, transform=None, target_transform=None, length=0):
        self.loaded_flag = False
        self.file = h5_file
        self.transform = transform
        self.target_transform = target_transform
        self.length = length
        self.gap = gap

    def __getitem__(self, index):
        if not self.loaded_flag:
            # print_memory_info('before load new h5 file...')
            with h5py.File(self.file, "r") as fh:
                print('PID {0}: read file {1}...'.format(os.getpid(), self.file))
                self.images = fh["images"][()]
                self.labels = fh["labels"][()]
                self.labels.resize((self.length,))
                self.loaded_flag = True
            # print_memory_info('after load new h5 file...')

        img = self.images[index]
        lab = self.labels[index]
        if self.transform is not None:
            img = self.transform(Image.fromarray(img))
        if self.target_transform is not None:
            lab = self.target_transform(lab)

        if index >= self.length - 1:
            # print_memory_info('before remove h5 file (del self.images, labels ...)...')
            self.loaded_flag = False
            del self.images
            del self.labels
            del self.length
            # print_memory_info('after remove h5 file (del self.images, labels ...)...')
        
        return img, lab

    def __len__(self):
        return self.length
