from threading import Timer
from base.common import get_cpu_percent, get_current_pc_name
from base.distributed_queue import DistributedQueue
import threading
import time
import numpy as np
from multiprocessing import Manager


class SchedularWorker:
    def __init__(self, worker_count, storage_adds, cal_rank):
        self.worker_count = worker_count
        # timer = threading.Timer(1, self._monitor_transform_time)
        cal_rank = cal_rank % len(storage_adds)
        self.transform_info_queue = DistributedQueue('cpu_info', storage_adds[str(cal_rank)], 9002, 'cpu_info', 300, False)
        self.tf_status = 0
        # self.time_storage_list = Manager().list([0] * 100)
        # self.time_compute_list = Manager().list([0] * 100)
        
        self.time_storage_list = [0] * 10
        self.time_compute_list = [0] * 10
        # timer.setDaemon(True)
        # timer.start()
        
    def _monitor_transform_time_NEW(self, time_storage_list, time_compute_list):
        average_storage_time = sum(time_storage_list) / len(time_storage_list)
        average_compute_time = sum(time_compute_list) / len(time_compute_list)
        # print('calculate average transform time in compute side. | success storage_time:{0} and compute_time:{1}'.format(average_storage_time, average_compute_time))
        for _ in range(self.worker_count):
            self.transform_info_queue.put({
                'host': get_current_pc_name(),
                'cpu_percent': None,
                'tf_status': self.tf_status,
                'average_storage_time': average_storage_time,
                'average_compute_time': average_compute_time
            })
    
    
    def _monitor_transform_time(self):
        average_storage_time = sum(self.time_storage_list) / len(self.time_storage_list)
        average_compute_time = sum(self.time_compute_list) / len(self.time_compute_list)
        # print('calculate average transform time in compute side. | success storage_time:{0} and compute_time:{1}'.format(average_storage_time, average_compute_time))
        for _ in range(self.worker_count):
            self.transform_info_queue.put({
                'host': get_current_pc_name(),
                'cpu_percent': None,
                'tf_status': self.tf_status,
                'average_storage_time': average_storage_time,
                'average_compute_time': average_compute_time
            })
        timer = threading.Timer(1, self._monitor_transform_time)
        timer.setDaemon(True)
        timer.start()

    def _monitor_cpu(self):
        cpu_percent = get_cpu_percent()
        cpu_info = {
            'host': get_current_pc_name(),
            'cpu_percent': cpu_percent,
            'tf_status': self.tf_status,
            'average_storage_time': None,
            'average_compute_time': None
        }

        for _ in range(self.worker_count):
            self.transform_info_queue.put(cpu_info)
    
    def set_transform_status(self, status):
        self.tf_status = status
    
    def set_transform_time(self, time_in_storage, time_in_compute):
        self.time_storage_list.append(time_in_storage)
        self.time_compute_list.append(time_in_compute)
        self.time_storage_list.pop(0)
        self.time_compute_list.pop(0)

        time_storage_list_new = np.array(self.time_storage_list)
        include0Cluster = sum(time_storage_list_new == 0)
        if include0Cluster == 0:
            self._monitor_transform_time_NEW(self.time_storage_list, self.time_compute_list)
        

