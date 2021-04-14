import socket
import struct
import pickle
from multiprocessing import Process, Queue


class NetWorker:
    def __init__(self, data_queue):
        self.data_queue = data_queue
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self._get_host_ip(), 9900))
        self.server.listen(5)
    
    def start_working(self):
        conn, addr = self.server.accept()
        self.conn = conn
        print(conn, addr)
        fetch_data_worker = Process(target=self.fetch_data)
        fetch_data_worker.daemon = True
        self.fetch_data_worker = fetch_data_worker
        fetch_data_worker.start()
    
    
    def _get_host_ip(self):
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)

    
    def fetch_data(self):
        while True:
            imginfo_size = struct.calcsize('l')
            img_info = self.conn.recv(imginfo_size)
            if img_info:
                img_size, = struct.unpack('l', img_info)
                recv_size = 0
                img_data = b''
                # if the server is on windows, and client is on linux(e.g. on hec), it need to uncomment the next line, or it could not fetch the data correctly.
                # data = self.conn.recv(4)
                while not recv_size == img_size:
                    if img_size - recv_size > 1024:
                        data = self.conn.recv(1024)
                    else:
                        data = self.conn.recv(img_size - recv_size)
                    recv_size += len(data)
                    img_data += data

                self.data_queue.put(pickle.loads(img_data))
    

    def stop_working(self):
        if self.conn:
            self.conn.close()
        self.fetch_data_worker.terminate()

