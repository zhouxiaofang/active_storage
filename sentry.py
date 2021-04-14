import numpy as np
import os
import shutil

# whereis_train = '/nfs/home/yfwang/imageNet/h5/no_compress_h5_many'
# whereis =       '/nfs/home/yfwang/imageNet/h5/no_compress_h5_many/train'

#whereis_train = '/mnt/orangefs/Imagenet/tars/h5'
#whereis = '/mnt/orangefs/Imagenet/tars'
whereis_train = '/mnt/orangefs/create_Big_h5Data'
whereis = '/mnt/orangefs/train_value_h5Data'
name = 'semaphore.npy'
file_name = os.path.join(whereis, name)

def get_slow():
    result = np.load(file_name)[0]
    return result

def get_fast():
    result = np.load(file_name)[1]
    return result

def add_slow():
    semaphore = np.load(file_name)
    semaphore[0] += 1
    np.save(file_name, np.array(semaphore))

def add_fast():
    semaphore = np.load(file_name)
    semaphore[1] += 1
    np.save(file_name, np.array(semaphore))

def canForward():
    slow = get_slow()
    fast = get_fast()
    if slow <= fast:
        return True
    return False

def set_top(top):
    semaphore = np.load(file_name)
    semaphore[2] = top
    np.save(file_name, np.array(semaphore))

def get_top():
    # return np.load(file_name)[2]
    result = np.load(file_name)[2]
    return result

def init():
    semaphore = np.load(file_name)
    semaphore[0] = 0
    np.save(file_name, np.array(semaphore))

def init_producer():
    semaphore = np.load(file_name)
    semaphore[1] = 0
    semaphore[2] = 1
    np.save(file_name, np.array(semaphore))

def kill_stale():
    for slow in range(get_slow()):
        if os.path.exists(os.path.join(whereis_train, 'train_' + str(slow))):
            shutil.rmtree(os.path.join(whereis_train, 'train_' + str(slow)))
