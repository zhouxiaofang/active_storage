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
import os
import random
import tarfile
from PIL import Image
import torchvision.transforms as transforms

import argparse
# 更改为HDF5格式写入 start
import h5py
import numpy as np
import glob
import sentry

DEFAULT_IMAGE_SIZE = 224
CHANNELS = 3
# 更改为HDF5格式写入 end
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--raw_data_dir',
                    help='path to dataset')

parser.add_argument('--local_scratch_dir',
                    help='path to store new dataset')

parser.add_argument('--shuffle_pool', default=1, type=int, metavar='N',
                    help='shuffle within ${sp} h5 files ')

TRAINING_SHARDS = 900
VALIDATION_SHARDS = 128

TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'val'

def make_shuffle_idx(n):
  order = list(range(n))
  random.shuffle(order)
  return order

def _check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.mkdir(directory)

def _is_png(filename):
    """Determine if a file contains a PNG format image.

    Args:
      filename: string, path of the image file.

    Returns:
      boolean indicating if the image is a PNG.
    """
    # File list from:
    # https://github.com/cytsai/ilsvrc-cmyk-image-list
    return 'n02105855_2933.JPEG' in filename

def _is_cmyk(filename):
    """Determine if file contains a CMYK JPEG format image.

    Args:
      filename: string, path of the image file.

    Returns:
      boolean indicating if the image is a JPEG encoded with CMYK color space.
    """
    # File list from:
    # https://github.com/cytsai/ilsvrc-cmyk-image-list
    blacklist = set(['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
                     'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
                     'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
                     'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
                     'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
                     'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
                     'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
                     'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
                     'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
                     'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
                     'n07583066_647.JPEG', 'n13037406_4650.JPEG'])
    return os.path.basename(filename) in blacklist

def _process_image(filename):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    #if filename.find("899") > -1:
    #    return np.ones((991, 224, 224, 3), dtype=np.int16), np.ones((991,))
    #return np.ones((1424, 224, 224, 3), dtype=np.int16), np.ones((1424,))
    f = h5py.File(filename, 'r')
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    data = f['images'][()]

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    #    transforms.ToTensor(),
    #    normalize,
    ])
    data = [np.array(transform(Image.fromarray(ele))) for ele in data]
    #return f['images'][()].map(Image.fromarray).map(transforms).map(np.array), f['labels'][()]
    return np.array(data), np.array(f['labels'][()])

def _process_image_files_batch(filenames, lengths, output_files):
    """Processes and saves list of images as HDF5Records.

    Args:
      filenames: list of strings; each string is a path to an image file
      output_files: string, unique identifier specifying the data set
      labels: map of string to integer; id for all synset labels
    """
    image_dbs, label_dbs = [], []

    '''
    这里需要 2*N*sizeof(h5) 倍内存占用，可以更加精细地优化~~~
    '''
    for filename in filenames:
        images, labels = _process_image(filename)
        image_dbs.append(images)
        label_dbs.append(labels)

    def get_detail(index):
        target = 0
        for ind in lengths:
          if index < ind:
            break
          index -= ind
          target += 1

        img = image_dbs[target][index]
        lab = label_dbs[target][index]
        return img, lab

    shuffle_idxs = make_shuffle_idx(np.sum(lengths))
    out_images, out_labels = [], []
    len_idx = 0
    for idx in shuffle_idxs:
        img, lab = get_detail(idx)
        out_images.append(img)
        out_labels.append(lab)
        if len(out_images) == lengths[len_idx]:
            f = h5py.File(output_files[len_idx], 'w')
            images_shape = (len(out_images), DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3)
            labels_shape = (len(out_labels),)
            f.create_dataset('images', shape=images_shape, maxshape=images_shape, data=out_images,
                             chunks=(1, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3))
            f.create_dataset('labels', shape=labels_shape, maxshape=labels_shape, dtype=np.dtype(np.int64), data=out_labels)
            f.close()
            out_images = []
            out_labels = []
            len_idx += 1



def _process_dataset(filenames, lengths, output_directory, shuffle_pool):
    """Processes and saves list of images as HDF5 file.

    Args:
      filenames: list of strings; each string is a path to an image file
      lengths: list of int; each int is the length to the file in filenames with same index
      output_directory: path where output files should be created
      num_shards: number of chucks to split the filenames into

    Returns:
      files: list of h5-record filepaths created from processing the dataset.
    """
    _check_or_create_dir(output_directory)

    files = []
    for shard in range(math.ceil((float)(len(filenames)) / shuffle_pool)):
        files_in_shuffle_pool = filenames[shard * shuffle_pool: (shard + 1) * shuffle_pool]
        files_lengths = lengths[shard * shuffle_pool: (shard + 1) * shuffle_pool]
        output_files = [os.path.join(
          output_directory, 'train-%.5d-of-%.5d.h5' % (idx, len(filenames))) for idx in range(shard * shuffle_pool, (shard + 1) * shuffle_pool)]
        _process_image_files_batch(files_in_shuffle_pool, files_lengths, output_files)

        files.extend(files_in_shuffle_pool)

        for file_name in files_in_shuffle_pool:
          print('Finished writing file: %s' % file_name)

    np.save(os.path.join(output_directory, "length.npy"), np.array(lengths))
    return files


def convert_to_h5_records(raw_data_dir, local_scratch_dir, shuffle_pool):
    """Convert the Imagenet dataset into H5-Record dumps."""

    # Shuffle training records to ensure we are distributing classes
    # across the batches.
    # random.seed(0)

    # Glob all the training files
    training_files = sorted(glob.glob(os.path.join(raw_data_dir, TRAINING_DIRECTORY, '*.h5')))
    training_length = np.load(os.path.join(raw_data_dir, TRAINING_DIRECTORY, 'length.npy'))
    print('所有的训练数据数量：{}'.format(np.sum(training_length)))

    # 生成next-generation H5Record时，组间shuffle，组内shuffle
    training_shuffle_idx = make_shuffle_idx(len(training_files))
    training_files = [training_files[i] for i in training_shuffle_idx]
    training_length = [training_length[i] for i in training_shuffle_idx]

    # Create training data

    print('Processing the training data.')
    training_records = _process_dataset(training_files, training_length, os.path.join(local_scratch_dir, TRAINING_DIRECTORY + '_' + str(sentry.get_fast() + 1)), shuffle_pool)
    print('train_filepath_h5record_len:{0}'.format(len(training_records)))

    return training_records


def main():  # pylint: disable=unused-argument
    args = parser.parse_args()
    local_scratch_dir = args.local_scratch_dir
    if local_scratch_dir is None:
        raise ValueError('Scratch directory path must be provided.')

    # Download the dataset if it is not present locally
    raw_data_dir = args.raw_data_dir
    if raw_data_dir is None:
        raise ValueError('raw_data_dir path must be provided.')

    #   first epoch use train dir
    # while sentry.get_fast() < sentry.get_top() - 1 and sentry.get_fast() < sentry.get_slow() + 1:
    # just for debug
    sentry.init_producer()

    while sentry.get_fast() < sentry.get_top() - 1:
        # Convert the raw data into h5-records
        convert_to_h5_records(raw_data_dir, local_scratch_dir, args.shuffle_pool)
        sentry.add_fast()
        # 如果produce超过consumer n个及以上的epoch，则执行删除操作
        if sentry.get_fast() > sentry.get_slow():
            sentry.kill_stale()

    print("next generation HDF5Record produced")

if __name__ == '__main__':
    main()
