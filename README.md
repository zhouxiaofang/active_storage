# README #
### What is this repository for?

数据集读取优化软件HDF5Record + 主动存储

### 常规使用命令:
使用形如以下命令将raw_data_dir目录下ImageNet数据集转化为.h5形式  
```bash  
python3 py_imagenet_to_h5s.py \
--raw_data_dir=/nfs/home/Imagenet_ILSVRC2012/ \
--local_scratch_dir=/nfs/home/yfwang/imageNet/h5/no_compress_h5_many/
```

使用形如以下命令，跑HDF5Record为数据集读取API的训练任务  
```bash
python3 main_h5_active.py -a resnet50 -j 1  --epochs 5 /nfs/home/yfwang/imageNet/h5/no_compress_h5_many/
```

使用形如以下命令，基于raw_data_dir目录下的train数据集，在local_scratch_dir目录下生成下一批次的train数据集  
```bash
python3 h5_next_producer.py --raw_data_dir=/nfs/home/yfwang/imageNet/h5/no_compress_h5_many/ --local_scratch_dir=/nfs/home/yfwang/imageNet/h5/no_compress_h5_many/ --shuffle_pool=10  
```
PS：shuffle_pool=10 参数表示，会随机选取10个.h5文件，在这10个.h5文件内全局shuffle，随机重新生成10个.h5文件；即为局部shuffle的pool大小
shuffle_pool=${所有.h5文件数量}是最理想的，但是数据集大小可能无法完全放入内存中，所以做了这个适配


### 文件详细介绍:
#### sentry.py为main_h5_active.py与h5_next_producer.py通信的简单模块：  
通过共享文件${whereis} + ${name} （/nfs/home/yfwang/imageNet/h5/no_compress_h5_many/train/semaphore.npy）  
1. 记录一个训练指针slow，生成指针fast；  
2. 并提供了一系列指针的操作接口。

#### h5_next_producer.py为主动存储模块的简单实现：
1. 主流程为一个while循环，结束条件为sentry.fast指针到达top，代表该训练所需的所有.h5批次都已经完成，结束任务。  
2. 还可通过参数--allowed-ahead控制生成.h5文件批次的脚步，只允许提前生成allowed-ahead批次的.h5文件。防止占用过多不必要的存储空间。  
3. shuffle_pool=10 参数表示，会随机选取10个.h5文件，在这10个.h5文件内全局shuffle，随机重新生成10个.h5文件；即为局部shuffle的pool大小。shuffle_pool=${所有.h5文件数量}是最理想的，但是数据集大小可能无法完全放入内存中，所以做了这个适配。  
4. 在main_h5_active.py在训练节点启动后，h5_next_producer.py需要在存储节点手动启动。

#### main_h5_active.py主要为模型训练相关逻辑，做了以下几个更改：
1. 增加了sentry模块相关的fast指针修改操作。  
2. 将ImageNet对应的数据集操作API ImageFolder替换为了HDF5Record。

#### H5Lib目录下文件，主要使用的是H5.py文件：
1. 其中的HDF5Record类继承pytorch的抽象类Dataset，需要实现其中的__getitem__，__lens__方法。  
2. 其中__lens__方法需要返回数据集的总样本个数。__getitem__方法根据\[0, __lens__())范围内的index，请求具体的样本。  
3. 具体由两层class实现，第一层class为HDF5Record，管理请求的index到对应的.h5文件的映射。对HDF5Dataset发起index2的请求。  
4. 另一层为HDF5Dataset，管理请求的index2到.h5文件内具体数据的映射。

#### h5_test.py和image_folder_test.py
提供了分别面向HDF5Record和ImageFolder的读取性能测试脚本。

#### H5_all_raw.py，H5_one_raw.py面向存储ImageNet数据集的原始图片（没有统一尺寸规格）的.h5，HDF5读取性能降低，存储体积成倍增大。为历史文件，不会使用：
1. 其中H5_all_raw.py为on_group的读取方式，一次性读取所需样本的整个.h5文件。  
2. H5_one_raw.py为on_demand的读取方式，仅仅读取所在.h5文件中所需样本的数据。




