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

#TRAINING_SHARDS = 900
# TRAINING_SHARDS = 100
TRAINING_SHARDS = 5
#VALIDATION_SHARDS = 128
# VALIDATION_SHARDS = 64
VALIDATION_SHARDS = 2

TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'validation'
# VALIDATION_DIRECTORY = 'val'


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
    image_data = Image.open(filename).convert('RGB')
    #    image_data = cv2.imread(filename)
    #    image_data = cv2.resize(image_data, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    #    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    image_data = np.array(transform(image_data))

    return image_data, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE

def _process_image_files_batch(output_file, filenames, synsets, labels):
    """Processes and saves list of images as HDF5Records.

    Args:
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
        output_file: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to an image file
        synsets: list of strings; each string is a unique WordNet ID
        labels: map of string to integer; id for all synset labels
    """
    images, heights, widthes, labellist, filelist, synsetlist, colorspaces, channels, formats = [], [], [], [], [], [], [], [], []
    # 这里有一点值得注意，resnet在write的时候，并没有将图片统一格式。而是在取出、预处理的时候统一了大小
    colorspace = b'RGB'
    channels = 3
    # image_format = b'JPEG'
    image_format = b'JPG'

    for filename, synset in zip(filenames, synsets):
        image_buffer, height, width = _process_image(filename)
        label = labels[synset]
        filename = filename.encode()

        images.append(image_buffer)
        # heights.append(height)
        # widthes.append(width)
        labellist.append(label)
        # filelist.append(filename)
        # synsetlist.append(synset)
        # colorspaces.append(colorspace)
        # channels.append(channel)
        # formats.append(image_format)

    # images = np.array(images)
    # heights = np.array(heights)
    # widthes = np.array(widthes)
    # labellist = np.array(labellist)
    # filelist = np.array(filelist)
    # synsetlist = np.array(synsetlist)
    # colorspaces = np.array(colorspaces)
    # channels = np.array(channels)
    # formats = np.array(formats)

    f = h5py.File(output_file, 'w')
    images_shape = (len(images), DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3)
    labels_shape = (len(labellist),)
    f.create_dataset('images', shape=images_shape, maxshape=images_shape, data=images, chunks=(1, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3))
    f.create_dataset('labels', shape=labels_shape, maxshape=labels_shape, dtype=np.dtype(np.int64), data=labellist)

    f.close()
    # writer.close()
    # 更改为HDF5格式写入 end


def _process_dataset(filenames, synsets, labels, output_directory, prefix,
                                         num_shards):
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
    chunksize = int(math.ceil(len(filenames) / num_shards))
    package_size_info = []
    files = []
    for shard in range(num_shards):
        chunk_files = filenames[shard * chunksize : (shard + 1) * chunksize]
        chunk_synsets = synsets[shard * chunksize : (shard + 1) * chunksize]
        output_file = os.path.join(output_directory, '%s-%.5d-of-%.5d.h5' % (prefix, shard, num_shards))
        _process_image_files_batch(output_file, chunk_files, chunk_synsets, labels)
        print('Finished writing file: %s' % output_file)
        files.append(output_file)
        package_size_info.append(len(chunk_files))

    update_h5_package_info(os.path.join(output_directory, 'length.npy'), package_size_info)
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

def convert_to_h5_records(raw_data_dir):
    """Convert the Imagenet dataset into H5-Record dumps."""

    # Shuffle training records to ensure we are distributing classes
    # across the batches.
    # random.seed(0)
    def make_shuffle_idx(n):
        order = list(range(n))
        random.shuffle(order)

        return order

    # Glob all the training files
    training_files = glob.glob(os.path.join(raw_data_dir, TRAINING_DIRECTORY, '*', '*.JPEG'))
    # training_files = glob.glob(os.path.join(raw_data_dir, TRAINING_DIRECTORY, '*', '*.JPG'))
    print('所有的训练数据数量：{}'.format(len(training_files)))


    training_shuffle_idx = make_shuffle_idx(len(training_files))
    training_files = [training_files[i] for i in training_shuffle_idx]

    # Get training file synset labels from the directory name
    training_synsets = [os.path.basename(os.path.dirname(f)) for f in training_files]
    # training_synsets = [training_synsets[i].encode() for i in training_shuffle_idx]

    # Glob all the validation files
    validation_files = sorted(glob.glob(os.path.join(raw_data_dir, VALIDATION_DIRECTORY, '*.JPEG')))
    # validation_files = sorted(glob.glob(os.path.join(raw_data_dir, VALIDATION_DIRECTORY, '*.JPG')))
    print('val数据量：{}'.format(len(validation_files)))
    # Get validation file synset labels from labels.txt
    validation_synsets = open(os.path.join(raw_data_dir, LABELS_FILE), 'rb').read().splitlines()
    #validation_synsets = [str(item) for item in validation_synsets]
    # Create unique ids for all synsets
    validation_synsets_strlist = [str(item) for item in validation_synsets]
    labels = {v: k for k, v in enumerate(sorted(set(validation_synsets_strlist + training_synsets)))}
    print('labels_总数量：{}'.format(len(labels)))
    # Create training data
    
    print('Processing the training data.')
    training_records = _process_dataset(
        training_files, training_synsets, labels,
        os.path.join(FLAGS.local_scratch_dir, TRAINING_DIRECTORY),
        TRAINING_DIRECTORY, TRAINING_SHARDS)
    print('train_filepath_h5record_len: {0}'.format(len(training_records)))
    # Create validation data
    print('Processing the validation data.')
    validation_records = _process_dataset(
        validation_files, validation_synsets_strlist, labels,
        os.path.join(FLAGS.local_scratch_dir, VALIDATION_DIRECTORY),
        VALIDATION_DIRECTORY, VALIDATION_SHARDS)
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
