import os
import random
import tensorflow as tf
import numpy as np
from pathlib import Path

def process_path(file_path, image_dimension, channels):
    """
    Load, decode, resize, and normalize an image file from disk.

    Parameters
    ----------
    file_path : tf.Tensor or str
        The path string targeting the image file to be processed.
    image_dimension : int
        The target width and height resolution for squaring the output image.
    channels : int
        The number of color channels to decode (e.g., 3 for RGB, 1 for Grayscale).

    Returns
    -------
    tf.Tensor
        A 3D float32 tensor representing the normalized image with values scaled
        within the range [0.0, 1.0].
    """
    img_raw = tf.io.read_file(file_path)
    img = tf.image.decode_png(img_raw, channels=channels)
    img = tf.image.resize(img, (image_dimension, image_dimension))
    img = tf.cast(img, tf.float32) / 255.0
    return img

def load_images(directory, image_dimension=256, channels=3, train_val_map: dict = {'train': 'train', 'val': 'val'}):
    """
    Construct training and validation tf.data.Dataset streams from a directory structure.

    This function scans specific subdirectories mapped by `train_val_map` for files
    matching the hardcoded pattern 'p*', shuffles the training paths, and maps the 
    `process_path` pipeline to parse and load images asynchronously.

    Parameters
    ----------
    directory : str or pathlib.Path
        The root directory containing the dataset splits.
    image_dimension : int, default 256
        The target resolution spatial width and height passed to the image processing pipeline.
    channels : int, default 3
        The expected number of structural channels for the decoded image tensors.
    train_val_map : dict, default {'train': 'train', 'val': 'val'}
        A dictionary mapping the keys 'train' and 'val' to their respective relative
        subdirectory string names inside the root directory.

    Returns
    -------
    train_mapped : tf.data.Dataset
        A parallelized dataset streaming unbatched training image tensors.
    val_mapped : tf.data.Dataset
        A parallelized dataset streaming unbatched validation image tensors.

    Raises
    ------
    AssertionError
        If `train_val_map` does not contain exactly two components or is missing 
        the explicit keys 'train' and 'val'.
    """
    assert (len(train_val_map) == 2) and ('train' in train_val_map) and ('val' in train_val_map), 'Train, validation mapping not valid!'
    
    directory = Path(directory)
    train_paths = list((directory / train_val_map['train']).glob('p*'))
    val_paths = list((directory / train_val_map['val']).glob('p*'))

    train_ds = tf.data.Dataset.from_tensor_slices([str(t) for t in train_paths])
    train_ds = train_ds.shuffle(buffer_size=len(train_paths), reshuffle_each_iteration=True)
    val_ds = tf.data.Dataset.from_tensor_slices([str(v) for v in val_paths])

    train_mapped = train_ds.map(lambda x: process_path(x, image_dimension, channels), num_parallel_calls=tf.data.AUTOTUNE)
    val_mapped = val_ds.map(lambda x: process_path(x, image_dimension, channels), num_parallel_calls=tf.data.AUTOTUNE)

    print(f"Loaded {len(train_paths)} images for training e {len(val_paths)} for validation.")
    return train_mapped, val_mapped
