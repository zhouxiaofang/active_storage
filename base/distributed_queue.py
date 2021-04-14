import queue, pickle
from multiprocessing.managers import BaseManager


class QueueManager(BaseManager):
    pass


class DistributedQueue():
    def __init__(self, queue_name, server_addrs, port, authkey, maxsize, server_queue=True): 
        cur_queue = queue.Queue(maxsize)
        self.q_manager = QueueManager(address=(server_addrs, port), authkey=bytes(authkey.encode('utf-8')))
        self.server_queue = server_queue
        if self.server_queue:
            QueueManager.register(queue_name, callable=lambda: cur_queue)
            self.q_manager.start()
            print('queue_manager start:', queue_name)
        else:
            QueueManager.register(queue_name)
            self.q_manager.connect()
        self.queue = eval('self.q_manager.{}()'.format(queue_name))
            

    def put(self, obj):
        self.queue.put(pickle.dumps(obj))

    def get(self, timeout=None):
        try:
            obj = self.queue.get(timeout=timeout)
        except queue.Empty:
            raise queue.Empty
    
        return pickle.loads(obj)

    def empty(self):
        return self.queue.empty()

    def shutdown(self):
        if self.server_queue:
            self.q_manager.shutdown()

    def qsize(self):
        return self.queue.qsize()
