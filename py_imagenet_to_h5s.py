r"""Script to download the Imagenet dataset and upload to gcs.

To run the script setup a virtualenv with the following libraries installed.
- `gcloud`: Follow the instructions on
    [cloud SDK docs](https://cloud.google.com/sdk/downloads) followed by
    installing the python api using `pip install gcloud`.
- `google-cloud-storage`: Install with `pip install google-cloud-storage`

Once you have all the above libraries setup, you should register on the
[Imagenet website](http://image-net.org/download-images) to get your
username and access_key.

Make sure you have around 300GB of disc space available on the machine where
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

from absl import app
from absl import flags

# 更改为HDF5格式写入 start
import h5py
import numpy as np
import glob
import torchvision.transforms as transforms
from PIL import Image
import tarfile
import socket
from base.common import fun_run_time

DEFAULT_IMAGE_SIZE = 224
CHANNELS = 3
# 更改为HDF5格式写入 end

flags.DEFINE_string(
        'project', None, 'Google cloud project id for uploading the dataset.')
flags.DEFINE_string(
        'local_scratch_dir', None, 'Scratch directory path for temporary files.')
flags.DEFINE_string(
        'raw_data_dir', None, 'Directory path for raw Imagenet dataset. '
        'Should have train and validation subdirectories inside it.')

FLAGS = flags.FLAGS

LABELS_FILE = 'synset_labels.txt'

TRAINING_CHUNKSIZES = 1000
VALIDATION_CHUNKSIZES = 1000

TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'val'
HOST_NAME = None
RANK = None

table = {
    'hec04': 0,
    'hec05': 1,
    'hec06': 2,
    'hec07': 3,
    'hec08': 4,
    'hec09': 5,
    'hec10': 6,
    'hec11': 7
}

def get_current_pc_name():
    global HOST_NAME, RANK
    HOST_NAME = socket.gethostname()
    RANK = table[HOST_NAME]

get_current_pc_name()

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


def _get_bytes_from_image(image_path):
    bytes_content = None
    with open(image_path, 'rb') as f:
        bytes_content = f.read()
    return bytes_content


def _package_images_batch(output_file, files, synsets, labels):
    int_labels = []
    images = []
    for file_info, synset in zip(files, synsets):
        cur_label = labels[synset]
        cur_image_buffer = _get_bytes_from_image(file_info)
        int_labels.append(cur_label)
        images.append(cur_image_buffer)

    with h5py.File(output_file, 'w') as hf:
        images_shape = (len(images),)
        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
        dset_images = hf.create_dataset('images', shape=images_shape, dtype=dt)
        dset_labels = hf.create_dataset('labels', shape=images_shape, dtype='uint16')
        
        for index, image_buffer in enumerate(images):
            dset_images[index] = list(image_buffer)
            dset_labels[index] = int_labels[index]


def _process_dataset(filenames, synsets, labels, output_directory, prefix, chunksize):
    """Processes and saves list of images as HDF5 file.

    Args:
        filenames: list of strings; each string is a path to an image file
        synsets: list of strings; each string is a unique WordNet ID
        labels: map of string to integer; id for all synset labels
        output_directory: path where output files should be created
        prefix: string; prefix for each file
        num_shards: number of chucks to split the filenames into

    Returns:
        files: list of h5-record filepaths created from processing the dataset.
    """
    _check_or_create_dir(output_directory)

    num_shards = int(math.ceil(len(filenames) / chunksize))
    package_size_info = []
    files = []
    for shard in range(num_shards):
        chunk_files = filenames[shard * chunksize : (shard + 1) * chunksize]
        chunk_synsets = synsets[shard * chunksize : (shard + 1) * chunksize]
        h5_output_file = os.path.join(output_directory, '%s-%s-%.5d-of-%.5d_%.6d.h5' % (HOST_NAME, prefix, shard, num_shards, len(chunk_files)))
        _package_images_batch(h5_output_file, chunk_files, chunk_synsets, labels)
        print('Finished writing file: {0}'.format(h5_output_file))
        files.append(h5_output_file)
        package_size_info.append(len(chunk_files))

    update_h5_package_info(os.path.join(output_directory, '{0}_length.npy'.format(HOST_NAME)), package_size_info)
    return files

def update_h5_package_info(path, info):
    """
        Save the package size in the length.npy file.
        Content like this: [100,  100,  100,  ..., 9]
               the size of  1.h5, 2.h5, 3.h5, ..., n.h5
        it's used for the h5_next_producer.py to get the all the h5 package size info.
    """
    if os.path.exists(path):
        os.remove(path)
    
    np.save(path, np.array(info))

@fun_run_time
def convert_to_h5_records(raw_data_dir):
    """Convert the Imagenet dataset into H5-Record dumps."""

    # Shuffle training records to ensure we are distributing classes
    # across the batches.
    random.seed(0)
    def make_shuffle_idx(n):
        order = list(range(n))
        random.shuffle(order)

        return order

    # Glob all the training files
    training_files = glob.glob(os.path.join(raw_data_dir, TRAINING_DIRECTORY, '*', '*.JPEG'))
    # training_files = glob.glob(os.path.join(raw_data_dir, TRAINING_DIRECTORY, '*', '*.JPG'))
    print('所有的训练数据数量：{}'.format(len(training_files)))

    training_synsets = [os.path.basename(os.path.dirname(f)) for f in training_files]

    # Glob all the validation files
    validation_files = glob.glob(os.path.join(raw_data_dir, VALIDATION_DIRECTORY, '*', '*.JPEG'))
    # validation_files = sorted(glob.glob(os.path.join(raw_data_dir, VALIDATION_DIRECTORY, '*.JPG')))
    print('val数据量：{}'.format(len(validation_files)))
    # Get validation file synset labels from labels.txt
    # validation_synsets = open(os.path.join(raw_data_dir, VALIDATION_DIRECTORY, LABELS_FILE), 'rb').read().splitlines()
    validation_synsets = [os.path.basename(os.path.dirname(f)) for f in validation_files]
    # Create unique ids for all synsets
    # validation_synsets_strlist = [str(item) for item in validation_synsets]
    labels = {v: k for k, v in enumerate(sorted(set(validation_synsets + training_synsets)))}
    print('labels_总数量：{}'.format(len(labels)))
    
    loop_time = 0
    cur_host_training_files = []
    while training_files[(loop_time * 8 + RANK) * TRAINING_CHUNKSIZES: (loop_time * 8 + RANK + 1) * TRAINING_CHUNKSIZES]:
        cur_host_training_files.extend(training_files[(loop_time * 8 + RANK) * TRAINING_CHUNKSIZES: (loop_time * 8 + RANK + 1) * TRAINING_CHUNKSIZES])
        loop_time += 1
    
    cur_host_training_synsets = [os.path.basename(os.path.dirname(f)) for f in cur_host_training_files]

    print('Processing the training data.')
    training_records = _process_dataset(
        cur_host_training_files, cur_host_training_synsets, labels,
        os.path.join(FLAGS.local_scratch_dir, TRAINING_DIRECTORY),
        TRAINING_DIRECTORY, TRAINING_CHUNKSIZES)
    print('train_filepath_h5record_len: {0}'.format(len(training_records)))
    # Create validation data
    print('Processing the validation data.')

    loop_time = 0
    cur_host_val_files = []
    while validation_files[(loop_time * 8 + RANK) * VALIDATION_CHUNKSIZES: (loop_time * 8 + RANK + 1) * VALIDATION_CHUNKSIZES]:
        cur_host_val_files.extend(training_files[(loop_time * 8 + RANK) * VALIDATION_CHUNKSIZES: (loop_time * 8 + RANK + 1) * VALIDATION_CHUNKSIZES])
        loop_time += 1
    
    cur_host_val_synsets = [os.path.basename(os.path.dirname(f)) for f in cur_host_val_files]

    validation_records = _process_dataset(
        cur_host_val_files, cur_host_val_synsets, labels,
        os.path.join(FLAGS.local_scratch_dir, VALIDATION_DIRECTORY),
        VALIDATION_DIRECTORY, VALIDATION_CHUNKSIZES)
    print('vali_filepath_h5record_len: {0}'.format(len(validation_records)))
    return training_records, validation_records


def main(argv):    # pylint: disable=unused-argument
    if FLAGS.local_scratch_dir is None:
        raise ValueError('Scratch directory path must be provided.')

    # Download the dataset if it is not present locally
    raw_data_dir = FLAGS.raw_data_dir
    if raw_data_dir is None:
        raw_data_dir = os.path.join(FLAGS.local_scratch_dir, 'raw_data')
        print('Downloading data to raw_data_dir: %s' % raw_data_dir)
        # download_dataset(raw_data_dir)

    # Convert the raw data into h5-records
    training_records, validation_records = convert_to_h5_records(raw_data_dir)
    return 0
    

if __name__ == '__main__':
    app.run(main)
