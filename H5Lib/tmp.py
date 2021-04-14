import numpy as np

def _re_mapping_index(index):
    h5_package_size = 100 
    batch_size = 10
    worker_count = 2

    h5_size = h5_package_size * batch_size
    big_batch_size = h5_size * worker_count

    current_remainder = index % batch_size
    worker_id = (int(index / batch_size)) % worker_count
    batch_idx = int((int(index / batch_size) - worker_id) / worker_count)
    h5_index = int(batch_idx / h5_package_size) * h5_size * worker_count + worker_id * h5_size + (batch_idx % h5_package_size) * batch_size + current_remainder

    return h5_index

a = _re_mapping_index(1267959)

len_array = np.load('C:/Users/DELL/Desktop/resnet50_test_result/length.npy')
print(a)
print(len_array)
print(len(len_array))