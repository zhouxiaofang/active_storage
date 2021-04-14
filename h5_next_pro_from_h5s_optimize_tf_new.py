"""
Make sure you have around 187GB * 2 of disc space available on the machine where
you're running this script. You can run the script using the following command.
```
python imagenet_to_gcs.py \
  --project="TEST_PROJECT" \
  --gcs_output_path="gs://TEST_BUCKET/IMAGENET_DIR" \
  --local_scratch_dir="./imagenet" \
  --imagenet_username=FILL_ME_IN \
  --imagenet_access_key=FILL_ME_IN \
```

Optionally if the raw data has already been downloaded you can provide a direct
`raw_data_directory` path. If raw data directory is provided it should be in
the format:
- Training images: train/n03062245/n03062245_4620.JPEG
- Validation Images: validation/ILSVRC2012_val_00000001.JPEG
- Validation Labels: synset_labels.txt
"""

import math
import os, io
import random
import tarfile
from PIL import Image
import torchvision.transforms as transforms
import json

import argparse
# 更改为HDF5格式写入 start
import h5py
import numpy as np
import glob
import sentry
import tarfile
import time
import gevent
import asyncio
import threading
import pickle
import queue
from base.common import fun_run_time, get_cpu_percent
from base.distributed_queue import DistributedQueue
from NetData.net_data import server_tf_state_table, client_tf_state_table
from multiprocessing import Process, Queue, Pool, Manager, Lock as p_Lock


H5_PACKAGE_SIZE = 1000 
DEFAULT_IMAGE_SIZE = 224
CHANNELS = 3

LOAD_WORKER_COUNT = 1
TRANSFORM_WORKER_COUNT = 1
WRITE_BACK_WORKER_COUNT = 1

image_queue = Queue(5 * H5_PACKAGE_SIZE)
transform_queue = Queue(5 * H5_PACKAGE_SIZE)
 

# 更改为HDF5格式写入 end
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--raw_data_dir', help='path to dataset')
parser.add_argument('--local_scratch_dir', help='path to store new dataset')
parser.add_argument('--shuffle_pool', default=1, type=int, metavar='N', help='shuffle within ${sp} h5 files ')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')

LOCAL_SCRATCH_DIR = None
RAW_DATA_DIR = None
SERVERS_CONFIG_FILE = '/nfs/home/zfang/storage_servers.json'

def get_storage_server_lists(servers_config_file):
    with open(servers_config_file, 'r') as f:
        storage_server_datas = json.loads(f.read())
        storage_server_lists = storage_server_datas['servers']
    return storage_server_lists

def make_shuffle_idx(n):
  order = list(range(n))
  random.shuffle(order)
  return order

def _check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.mkdir(directory)

class WorkerProcess(Process):
    def __init__(self, worker_pool):
        Process.__init__(self)
        self.finish_flag = False
        self.worker_pool = worker_pool
        
    def set_worker_finish(self, all_finish_flag):
        self.finish_flag = True
        self.worker_pool.remove_worker(self, all_finish_flag)


# load_images_to_memory
class LoadImagesWorker(WorkerProcess):
    def __init__(self, h5_data, shuffle_pool, image_queue, p_lock, worker_pool):
        WorkerProcess.__init__(self, worker_pool)
        self.h5_data = h5_data
        self.shuffle_pool = shuffle_pool
        self.image_queue = image_queue
        self.image_count = 0
        self.p_lock = p_lock
    
    def _extract_images(self, filename):
        images_bytes_list = None
        labels = None
        time1 = time.time()
        with h5py.File(filename, 'r') as hf:
            images_bytes_list = hf['images'][()]
            labels = hf['labels'][()]
        print('    extract one h5 file time used: | e_h5_{0}'.format(time.time() - time1))
        return images_bytes_list, labels
    
    def get_pool_data(self):
        result = []
        self.p_lock.acquire()
        fetch_count = 0
        if len(self.h5_data) < self.shuffle_pool:
            fetch_count = len(self.h5_data)
        else:
            fetch_count = self.shuffle_pool
        
        for _ in range(fetch_count):
            result.append(self.h5_data.pop(0))
        
        self.p_lock.release()
        return result

    def run(self):
        print('Load image worker start working...')
        Load_image_start_time = time.time()
        while not self.finish_flag:
            pool_data = self.get_pool_data()
            if not pool_data:
                # could get more h5 data
                self.set_worker_finish(all_finish_flag=True)
                continue
            self._process_image_files_batch(pool_data)
        print('Load image worker finished, cost_load_time%{}'.format(time.time() - Load_image_start_time))

    def _process_image_files_batch(self, files_in_shuffle_pool):
        shuffle_pool_images = []
        shuffle_pool_labels = []
        images_lengths = []
        for filename in files_in_shuffle_pool:
            images_lengths.append(int(os.path.basename(filename).split('_')[1].replace('.h5', '')))
            images_bytes_list, labels = self._extract_images(filename)
            shuffle_pool_images.extend(images_bytes_list)
            shuffle_pool_labels.extend(labels)
        
        training_shuffle_idx = make_shuffle_idx(len(shuffle_pool_images))
        shuffle_pool_images = [shuffle_pool_images[i] for i in training_shuffle_idx]
        shuffle_pool_labels = [shuffle_pool_labels[i] for i in training_shuffle_idx]
        for index, image in enumerate(shuffle_pool_images): # index, image in enumerate(),能够遍历索引index 0,1,2...
            self.image_count += 1
            # print('read and send {0}th image....'.format(self.image_count))
            self.image_queue.put({
                'image': image,
                'label': shuffle_pool_labels[index]
            })
        pass


# transform_image_from_queue
class TransformImagesWorker(WorkerProcess):
    def __init__(self, image_queue, transform_queue, distribute_image_queue, distribute_transform_info_queue, tf_img_worker_pool, ld_img_worker_pool):
        WorkerProcess.__init__(self, tf_img_worker_pool)
        self.image_queue = image_queue
        self.transform_queue = transform_queue
        self.ld_img_worker_pool = ld_img_worker_pool
        self.tf_status = 0
        self.cur_tf_status = self.tf_status
        self.distribute_image_queue = distribute_image_queue
        self.distribute_transform_info_queue = distribute_transform_info_queue

    def _process_image(self, image_data):
        """Process a single image file.

        Args:
            filename: string, path to an image file e.g., '/path/to/example.JPG'.
        Returns:
            image_buffer: string, JPEG encoding of RGB image.
            height: integer, image height in pixels.
            width: integer, image width in pixels.
        """
        start_time = time.time()
        self.cur_tf_status = self.tf_status
        is_tf_rgb = server_tf_state_table[self.cur_tf_status][0]
        # print('storage_client is_tf_rgb:', is_tf_rgb)
        if is_tf_rgb:
            image_buffer = io.BytesIO(bytes(image_data))
            image_data = Image.open(image_buffer).convert('RGB')

        transform_list = server_tf_state_table[self.cur_tf_status][1:]
        # print('transform_list ==>', transform_list)


        transform = transforms.Compose(transform_list)
        image_data = transform(image_data)
        time_used = time.time() - start_time
        print('storage_client transform one image cost_time+{}'.format(time_used))
        return image_data, time_used
    

    def transform_image_task(self):
        # print('Transforms image worker start working...')
        n = 0
        while not self.finish_flag:
            # print('before read image from image_queue, image_queue size: {0}...'.format(self.image_queue.qsize()))
            try:
                image_info = self.image_queue.get(timeout=5)
            except queue.Empty:
                if len(self.ld_img_worker_pool.task_not_finish_list) == 0:
                    # print('all_finish_flag is true, queue size: {0}...'.format(self.image_queue.qsize()))
                    # if there is no more worker in ld_img_worker_pool.task_not_finish_list and the current image_queue is empty, so it has no more thansform task to do.
                    self.set_worker_finish(all_finish_flag=True)
                    break
                continue
            
            # print('after process image from image_info of image_queue...')
            # if self.image_queue.qsize() > 0 and self.image_queue.qsize() % 100 == 0:
            #     print('image_queue get one batch, its size: {0}'.format(self.image_queue.qsize()))
                
            label = image_info['label']
            images_buffer, time_used = self._process_image(image_info['image'])

            self.distribute_image_queue.put({
                'image': images_buffer,
                'label': label,
                'tf_status': self.cur_tf_status,
                'time_used': time_used
            })
            n += 1
            # print('send {0}th image...'.format(n))
        
        self.distribute_image_queue.put('1')
        self.distribute_image_queue.put('1')

        # print('{0} transform images done! Removed from transform_images_done list... | qsize:{1}'.format(os.getpid(), self.distribute_image_queue.qsize()))

    def adjust_transform_status_task_by_time(self):
        while not self.finish_flag:
            tf_info = None
            try:
                tf_info = self.distribute_transform_info_queue.get(timeout=5)
            except queue.Empty:
                if self.finish_flag:
                    break
                continue
            
            cur_workers_tf_status = tf_info['tf_status']
            cur_average_storage_time = tf_info['average_storage_time']
            cur_average_compute_time = tf_info['average_compute_time']
            # print('get transform status task info...| current compute_time:{0} and storage_time:{1}'.format(cur_average_compute_time, cur_average_storage_time))
            percent_rate = 100
            if cur_average_storage_time == 0:
                pass
            else:
                percent_rate = (cur_average_compute_time + 0.003) / cur_average_storage_time
            # print("percent_rate cal success :", percent_rate)
            #  6/4 7/3 8/2 9/1 ...
            if percent_rate > 1.5:
                if self.tf_status >= cur_workers_tf_status and self.tf_status > 0:
                    self.tf_status -= 1
                    # print('success reduce tf_status, value: {0}'.format(self.tf_status))
            elif percent_rate < 0.6666:
                if self.tf_status <= cur_workers_tf_status and self.tf_status < 5:
                    self.tf_status += 1
                    # print('success increate tf_status, value: {0}'.format(self.tf_status))


    def monitor_queue_empty(self):
        while True:
            if self.distribute_image_queue.empty() and self.finish_flag:
                break
            # if the distribute_image_queue not empty, wait the consumer fetch its data.
            print('wait consumer consume the distribute_image_queue... | qsize==>', self.distribute_image_queue.qsize())
            if 1 <= self.distribute_image_queue.qsize() and self.distribute_image_queue.qsize() <= 2:
                break
            time.sleep(3)

    def run(self):
        t1 = threading.Thread(target=self.transform_image_task)
        t2 = threading.Thread(target=self.adjust_transform_status_task_by_time)
        t3 = threading.Thread(target=self.monitor_queue_empty)
        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()
        t3.join()
        print('TransformImagesWorker done!')


class WorkerPool():
    def __init__(self):
        self.p_lock = p_Lock()
        self.worker_list = []
        self.removed_worker_list = []
        self.all_finish_flag = False
        self.max_worker_count = 2

    def start_worker(self):
        for worker in self.worker_list:
            worker.start()
    
    def join_worker(self):
        for worker in self.worker_list:
            worker.join()

    def add_worker(self):
        pass
    
    def remove_worker(self):
        pass


class LoadImagesWorkerPool(WorkerPool):
    def __init__(self, worker_count, h5_data, shuffle_pool, image_queue):
        WorkerPool.__init__(self)
        self.worker_count = worker_count
        self.h5_data = h5_data
        self.shuffle_pool = shuffle_pool
        self.image_queue = image_queue
        self.task_not_finish_list = Manager().list()

        for _ in range(worker_count):
            load_image_worker = LoadImagesWorker(self.h5_data, self.shuffle_pool, self.image_queue, self.p_lock, self)
            load_image_worker.daemon = True
            self.worker_list.append(load_image_worker)
            self.task_not_finish_list.append(id(load_image_worker))
    
    def add_worker(self):
        if len(self.worker_list) == self.max_worker_count:
            return
    
        load_image_worker = LoadImagesWorker(self.h5_data, self.shuffle_pool, self.image_queue, self.p_lock, self)
        load_image_worker.daemon = True
        self.worker_list.append(load_image_worker)
        self.task_not_finish_list.append(id(load_image_worker))
    
    def remove_worker(self, worker=None, all_finish_flag=False):
        if len(self.worker_list) == 0:
            return
        
        if not worker:
            self.worker_list[-1].finish_flag = True
            worker = self.worker_list.pop(-1)
        self.removed_worker_list.append(worker)
        self.worker_list.remove(worker)
        self.task_not_finish_list.remove(id(worker))
        self.all_finish_flag = all_finish_flag


class TransformImagesWorkerPool(WorkerPool):
    def __init__(self, worker_count, image_queue, transform_queue, ld_img_worker_pool, queue_name, storage_server_addrs, node_num):
        WorkerPool.__init__(self)
        self.worker_count = worker_count
        self.image_queue = image_queue
        self.transform_queue = transform_queue
        self.ld_img_worker_pool = ld_img_worker_pool
        self.max_worker_count = 8
        self.task_not_finish_list = Manager().list()
        self.distribute_image_queue = DistributedQueue(queue_name, storage_server_addrs[str(node_num)], 9001, 'abcdef', 5*1000)
        self.distribute_transform_info_queue = DistributedQueue('cpu_info', storage_server_addrs[str(node_num)], 9002, 'cpu_info', 100)

        for _ in range(worker_count):
            tf_image_worker = TransformImagesWorker(self.image_queue, self.transform_queue, self.distribute_image_queue, self.distribute_transform_info_queue, self, self.ld_img_worker_pool)
            tf_image_worker.daemon = True
            self.worker_list.append(tf_image_worker)
            self.task_not_finish_list.append(id(tf_image_worker))
    
    def add_worker(self):
        if len(self.worker_list) > self.max_worker_count:
            return
        
        tf_image_worker = TransformImagesWorker(self.image_queue, self.transform_queue, self.distribute_image_queue, self.distribute_transform_info_queue, self, self.ld_img_worker_pool)
        tf_image_worker.daemon = True
        self.worker_list.append(tf_image_worker)
        self.task_not_finish_list.append(id(tf_image_worker))
    
    def remove_worker(self, worker=None, all_finish_flag=False):
        if len(self.worker_list) == 0:
            return
        
        if not worker:
            # self.worker_list[-1].set_worker_finish()
            self.worker_list[-1].finish_flag = True
            worker = self.worker_list.pop(-1)
        self.removed_worker_list.append(worker)
        self.worker_list.remove(worker)
        self.task_not_finish_list.remove(id(worker))
        self.all_finish_flag = all_finish_flag
    
    def __del__(self):
        self.distribute_image_queue.shutdown()
        self.distribute_transform_info_queue.shutdown()
        print('TransformImagesWorkerPool deconstruction!') # 自动销毁队列吗？？？


@fun_run_time
def main():  # pylint: disable=unused-argument
    global LOCAL_SCRATCH_DIR, RAW_DATA_DIR
    args = parser.parse_args()
    LOCAL_SCRATCH_DIR = args.local_scratch_dir
    if LOCAL_SCRATCH_DIR is None:
        raise ValueError('Scratch directory path must be provided.')

    # Download the dataset if it is not present locally
    RAW_DATA_DIR = args.raw_data_dir
    if RAW_DATA_DIR is None:
        raise ValueError('raw_data_dir path must be provided.')

	
    hec_rank = args.rank
    transform_queue_name = 'image_queue_' + str(hec_rank)
    server_addrs =  get_storage_server_lists(SERVERS_CONFIG_FILE)

    sentry.init_producer()
    # tf_time_epoch_start = time.time()
    while sentry.get_fast() < sentry.get_top():
        # Convert the raw data into h5-records
        one_epoch_start_time = time.time()

        training_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, 'train', '*.h5')))
        # training_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, '*.h5')))[:5]
        print('所有的训练数据数量：{}'.format(len(training_files)))
        
        node_nums = len(server_addrs)
        per_node_files_len = int(len(training_files) / node_nums)
        per_node_remain_len = int(len(training_files) % node_nums)
        per_node_start = 0
        for kth in range(node_nums):
            if kth < per_node_remain_len:
                per_node_files = training_files[per_node_start: (per_node_start + per_node_files_len + 1)]
                per_node_start += per_node_files_len + 1
            else:
                per_node_files = training_files[per_node_start: per_node_start + per_node_files_len]
                per_node_start += per_node_files_len

            if kth == hec_rank:
                break

        print('每个存储节点的训练数据的H5数量===>{}'.format(len(per_node_files)))
        # 生成next-generation H5Record时，组间shuffle，组内shuffle
        training_shuffle_idx = make_shuffle_idx(len(per_node_files))
        per_node_training_files = [per_node_files[i] for i in training_shuffle_idx]

        ld_worker_pool = LoadImagesWorkerPool(worker_count=LOAD_WORKER_COUNT, h5_data=per_node_training_files, shuffle_pool=args.shuffle_pool, image_queue=image_queue)
        tf_worker_pool = TransformImagesWorkerPool(worker_count=TRANSFORM_WORKER_COUNT, image_queue=image_queue, transform_queue=transform_queue, 
													ld_img_worker_pool=ld_worker_pool, queue_name=transform_queue_name, storage_server_addrs=server_addrs, node_num=hec_rank)
        # wb_worker_pool = WritebackImagesWorkerPool(worker_count=WRITE_BACK_WORKER_COUNT, transform_queue=transform_queue, tf_img_worker_pool=tf_worker_pool)
        ld_worker_pool.start_worker()
        tf_worker_pool.start_worker()
        # wb_worker_pool.start_worker()
        ld_worker_pool.join_worker()
        tf_worker_pool.join_worker()
        # wb_worker_pool.join_worker()
        print('===============')
        print('start next epoch!')
    
        # convert_to_h5_records(raw_data_dir, local_scratch_dir, args.shuffle_pool)
        sentry.add_fast()
        # 如果produce超过consumer n个及以上的epoch，则执行删除操作
        if sentry.get_fast() > sentry.get_slow():
            sentry.kill_stale()
        
        print('handle one epoch images total_time cost: {0}'.format(time.time() - one_epoch_start_time))

    # tf_time_epoch_end = time.time()
    # print("next generation HDF5Record produced | time_0:{0}".format((tf_time_epoch_end - tf_time_epoch_start)))

if __name__ == '__main__':
    main()
