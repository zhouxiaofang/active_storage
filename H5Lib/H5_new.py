import os.path
import torch.utils.data as data
import h5py
import glob
import time
import torch
from PIL import Image
import numpy as np
from multiprocessing import Pool

class HDF5Record(data.Dataset):
    def __init__(self, root, gap, batch_size, worker_count, h5_package_size, prefetch_factor=1, transform=None, target_transform=None):
        self.batch_size = batch_size
        self.worker_count = worker_count
        self.h5_package_size = h5_package_size
        self.dbs = []
        self.prefetch_factor = prefetch_factor
        
        h5_dirs = sorted(glob.glob(os.path.join(root, '*.h5')))
        len_array = np.load(os.path.join(root, 'length.npy'))
        self.biggest_index = sum(len_array) - 1
        index = 0
        self.initialized_pool = False
        self.prefetch_table = {}
        if self.worker_count > 1:
            self.no_map_index = self.biggest_index // (batch_size * h5_package_size * worker_count) * (batch_size * h5_package_size * worker_count)

        # Treat the dbs as sorted dbs by default.
        # because during the h5 packaging, it has already sorted the db names when generated it. If not, it will not accelerate the pre-loading speed as the next design.
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
        self.length = count
        
    @staticmethod
    def prefetch_h5(file_name, label_length):
        try:
            with h5py.File(file_name, "r") as fh:
                print('PID {0}: read file {1}...'.format(os.getpid(), file_name))
                images = fh["images"][()]
                labels = fh["labels"][()]
                labels.resize((label_length,))
        except Exception as ex:
            print('catch ex: {0}'.format(ex))
        
        return images, labels
    
    def _inital_process_pool(self):
        # Here inital a process pool which has prefetch_factor process(es)
        print('process {0} start inintialize process pool.'.format(os.getpid()))
        if self.prefetch_factor > 0:
            self.pool = Pool(self.prefetch_factor)
            self.initialized_pool = True
        
      
    def _re_mapping_index(self, index):
        ''' 
        Used for re-mapping input index
        Premise:
        1. use h5p package
        2. use more than 1 subProcess.
        3. if find one index which is the big_batch_size * N, and the index plus another big_batch_size is bigger than the biggest data index. Then don't re-mapp it.
            big_batch_size = h5_size * worker_count

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


    def __getitem__(self, index):
        if not self.initialized_pool:
            self._inital_process_pool()

        if self.worker_count > 1:
            index = self._re_mapping_index(index)

        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind
        
        # 用进程池加载预取的数据.
        if self.prefetch_factor:
            for item in range(self.prefetch_factor):
                prefetch_target = target + (item + 1) * self.worker_count
                if prefetch_target < len(self.dbs) and prefetch_target not in self.prefetch_table.keys():
                    print('Current target db: {0}, prefect target db: {1}'.format(target, prefetch_target))
                    prefecth_db = self.dbs[prefetch_target]
                    self.prefetch_table[prefetch_target] = self.pool.apply_async(self.prefetch_h5, args=(prefecth_db.file, prefecth_db.length,))
        
        db = self.dbs[target]
        if target in self.prefetch_table.keys():
            # 如果进程池已经去加载当前h5， 那么去查询是否线程池已经load完数据，如果没有完成，get方法会阻塞住，直到结果返回.
            cur_target_res = self.prefetch_table[target]
            images, labels = cur_target_res.get()
            db.set_value(images, labels)
            del cur_target_res
            del self.prefetch_table[target]
            pass
        else:
            # 如果进程池没有去加载当前h5，那么主进程去加载
            # 主进程只有刚开始的时候才需要加载属于自己的第一个h5文件，所以它不需要维护prefetch_table 
            pass

        # currently, in the case of using prefetch data.
        # the current db should has data to handle.
        index = index - sub
        img, lab = db[index]
        return img, lab

    def __len__(self):
        return self.length

class HDF5Dataset(data.Dataset):
    def __init__(self, h5_file, gap = 0, transform=None, target_transform=None, length=0):
        self.load_flag = False
        self.file = h5_file
        self.transform = transform
        self.target_transform = target_transform
        self.length = length
        self.gap = gap
        # self.env = None
        self.images = None
        self.labels = None
        self.load_flag = False

    def set_value(self, images, labels):
        self.images = images
        self.labels = labels
        self.load_flag = True

    def __getitem__(self, index):
        loop_count = 0
        if not self.load_flag:
            with h5py.File(self.file, "r") as fh:
                print('PID {0}: read file {1}...'.format(os.getpid(), self.file))
                self.images = fh["images"][()]
                self.labels = fh["labels"][()]
                self.labels.resize((self.length,))
                self.load_flag = True
        else:
            # here should not coming in...
            # because if the current db does not have data, it will loop in line 136
            while not self.load_flag:
                if loop_count % 10000 == 0:
                    print("looped {0} times\n".format(loop_count))
                loop_count += 1

        img = self.images[index]
        lab = self.labels[index]
        if self.transform is not None:
            img = self.transform(Image.fromarray(img))
        if self.target_transform is not None:
            lab = self.target_transform(lab)
        if index >= self.length - 1:
            del self.images
            del self.labels
            del self.length

        return img, lab

    def __len__(self):
        return self.length
