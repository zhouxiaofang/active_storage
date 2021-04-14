def _re_mapping_index_distribute(index):
    ''' 
    Used for re-mapping input index for distributed training model.
    Premise:
    1. at least 2 computing nodes
    2. use h5p package
    '''
    world_size = 2
    batch_size = 10
    h5_package_size = 4
    worker_count = 3

    
    gap_size = world_size * batch_size
    node_id = index % world_size
    h5_size = batch_size * h5_package_size
    # whole_size means the count of pictures it needed if each subProcess per node could get one h5 package data
    whole_size = h5_size * worker_count * world_size
    worker_id = index // gap_size % worker_count
    current_remainder = (index - node_id) % gap_size // world_size

    A = index // whole_size
    B = index % whole_size // (world_size * worker_count * batch_size)
    C = index // (world_size * worker_count * batch_size)

    h5_index = A * whole_size + node_id * h5_size + worker_id * h5_size * world_size + B * batch_size + current_remainder
    return h5_index


obj_list = [item for item in range(481)]
list1 = obj_list[0: -1: 2]
list2 = obj_list[1: -1: 2]

count = 0 
for item in list1:
    new_item = _re_mapping_index_distribute(item)
    print('item: {0}, new item:{1}'.format(item, new_item))
    count += 1
    if count % 10 == 0:
        print("=====================================")
        pass

count = 0 
for item in list2:
    new_item = _re_mapping_index_distribute(item)
    print('item: {0}, new item:{1}'.format(item, new_item))
    count += 1
    if count % 10 == 0:
        print("=====================================")
        pass
