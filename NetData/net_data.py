import os, io
import os.path
import torch.utils.data as data
import h5py
import glob
import time
import torch
from PIL import Image
import numpy as np
import socket
from .schedular_worker import SchedularWorker
from multiprocessing import Pool
import torchvision.transforms as transforms
from base.distributed_queue import DistributedQueue
import sys
import random


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
client_tf_state_table = {
    0: [False],
    1: [False, normalize],
    2: [False, transforms.ToTensor(), normalize],
    3: [False, transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize],
    4: [False, transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize],
    5: [True,  transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
}

server_tf_state_table = {
    0: [True, transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize],
    1: [True, transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()],
    2: [True, transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()],
    3: [True, transforms.RandomResizedCrop(224)],
    4: [True],
    5: [False]
}


class NetRecord(data.Dataset):
    def __init__(self, dataset_path, server_storage_addrs, cal_rank, per_node_train_imgs):
        self.server_storage_addrs = server_storage_addrs
        self.transforms_table = client_tf_state_table[0]
        h5_dirs = sorted(glob.glob(os.path.join(dataset_path, '*.h5')))
        length = 0
        for h in h5_dirs:
            cur_h5_size = int(os.path.basename(h).split('_')[1].replace('.h5', ''))
            length += cur_h5_size
        self.length = length
        self.schedular_worker = SchedularWorker(2 , server_storage_addrs, cal_rank)
        self.image_count = 0
        self.last_tf_status = 0
        self.rank = cal_rank
        self.storage_node_nums = len(server_storage_addrs)
        self.storage_queue_table = []
        storage_len =  len(server_storage_addrs)
        for kth_storage_num in range(storage_len):
            transform_queue_name = 'image_queue_' + str(kth_storage_num)
            distribute_data_queue = DistributedQueue(transform_queue_name, self.server_storage_addrs[str(kth_storage_num)], 9001, 'abcdef', 5*1000, False)
            self.storage_queue_table.append(distribute_data_queue)

    def get_image_data_info(self):
        global image_data_info
        try:
            kth_storage_num = random.randint(0, len(self.storage_queue_table) - 1)
            print('get_image_data_info  kth_storage_num value: {0}'.format(kth_storage_num))
            get_data_network_time = time.time()
            image_data_info = self.storage_queue_table[kth_storage_num].get()
            print('get_data_network_time&{0}'.format(time.time() - get_data_network_time))
            
            if isinstance(image_data_info, str):
                self.storage_queue_table.remove(self.storage_queue_table[kth_storage_num])
                image_data_info = self.get_image_data_info()     
        except Exception as ex:
            print('ex 11: {0}'.format(ex))
        return image_data_info


    def __getitem__(self, index):
        self.image_count += 1
        print('consum {0} image'.format(self.image_count))
        try:
            image_info = self.get_image_data_info()
        except Exception as ex:
            print('ex 22: {0}'.format(ex))
            
        label = image_info['label']
        image = image_info['image']
        tf_status = image_info['tf_status']
        time_used_in_storage = image_info['time_used']
        
        if self.last_tf_status != tf_status:
            self.set_transform(tf_status)
            self.schedular_worker.set_transform_status(tf_status)
            self.last_tf_status = tf_status
    
        if self.transforms_table:
            # print("cal client | success transform_id", self.last_tf_status)
            try:
                start_time = time.time()
                # print('cal_client | self.transforms_table ==>{0} and is_tf_rgb: {1}'.format(self.transforms_table, self.transforms_table[0]))
                if self.transforms_table[0]:
                    image_buffer = io.BytesIO(bytes(image))
                    image = Image.open(image_buffer).convert('RGB')

                transform_list = self.transforms_table[1:]
                transform_new = transforms.Compose(transform_list)
                # print("storage client | success transform_list :", transform_list)
                image = transform_new(image)
                
                time_used_in_compute = time.time() - start_time
                print('cal_node tf_one_pic_consume_time+{0}'.format(time_used_in_compute))
                self.schedular_worker.set_transform_time(time_used_in_storage, time_used_in_compute)
            except Exception as ex:
                print('ex: {0}'.format(ex))

        # note the label should be int, or if will have exception during default_collate.
        return np.array(image), int(label)


    def set_transform(self, transform_status):
        if transform_status not in client_tf_state_table.keys():
            print('transform status is not existing...')
            return 
        
        self.transforms_table = client_tf_state_table[transform_status]
    

    def __len__(self):
        return self.length

