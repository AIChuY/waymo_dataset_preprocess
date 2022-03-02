"""
Scripts to convert Waymo dataset into pictures.
"""

import time

import tensorflow as tf

import utils

FILENAME = (
    "/home/wzy/下载/uncompressed_tf_example_training_training_"
    "tfexample.tfrecord-00000-of-01000"
)


def main():
    """
    Testing image generation.
    """
    # Setting up data loader.
    features_description = utils.build_features_description()
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type="")
    iterator = dataset.as_numpy_iterator()
    tic = time.time()
    # Parsing raw data.
    for i in range(10):
        data = next(iterator)
        parsed = tf.io.parse_single_example(data, features_description)
        utils.render_img(parsed, "./%d/" % i)
        print(i)
    toc = time.time()
    print(toc - tic)


if __name__ == "__main__":
    main()
