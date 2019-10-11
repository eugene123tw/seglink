import os
import glob

import numpy as np
import tensorflow as tf
from skimage import io

# import util
from datasets.dataset_utils import convert_to_example


def cvt_to_tfrecords(output_path, data_path, gt_path):
    image_paths = glob.glob(data_path + '/*.jpg')
    print("%d images found in %s" % (len(image_paths), data_path))

    with tf.python_io.TFRecordWriter(output_path) as tfrecord_writer:
        for idx, image_path in enumerate(image_paths):
            oriented_bboxes = []
            bboxes = []
            labels = []
            labels_text = []
            ignored = []
            print("\tconverting image: %d/%d %s" % (idx, len(image_paths), image_path))
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()

            image = io.imread(image_path)
            shape = image.shape
            h, w = shape[0:2]
            h *= 1.0
            w *= 1.0
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            gt_name = 'gt_' + image_name + '.txt'
            gt_filepath = os.path.join(gt_path, gt_name)

            with open(gt_filepath, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                gt = line.split(",")
                oriented_box = [int(gt[i]) for i in range(8)]
                oriented_box = np.asarray(oriented_box) / ([w, h] * 4)
                oriented_bboxes.append(oriented_box)

                xs = oriented_box.reshape(4, 2)[:, 0]
                ys = oriented_box.reshape(4, 2)[:, 1]
                xmin = xs.min()
                xmax = xs.max()
                ymin = ys.min()
                ymax = ys.max()
                bboxes.append([xmin, ymin, xmax, ymax])
                ignored.append(gt[-1].find('###') >= 0)

                # might be wrong here, but it doesn't matter because the label is not going to be used in detection
                labels_text.append(gt[-1])
                labels.append(1)
            example = convert_to_example(image_data, image_name, labels, ignored, labels_text, bboxes, oriented_bboxes,
                                         shape)
            tfrecord_writer.write(example.SerializeToString())


if __name__ == "__main__":
    output_dir = '/home/eugene/_DATASETS/scene_text/icdar_2015'
    training_data_dir = '/home/eugene/_DATASETS/scene_text/icdar_2015/train'
    training_gt_dir = '/home/eugene/_DATASETS/scene_text/icdar_2015/train_gt'
    test_data_dir = '/home/eugene/_DATASETS/scene_text/icdar_2015/test'
    test_gt_dir = '/home/eugene/_DATASETS/scene_text/icdar_2015/test_gt'

    cvt_to_tfrecords(output_path=os.path.join(output_dir, 'icdar2015_train.tfrecord'),
                     data_path=training_data_dir,
                     gt_path=training_gt_dir)
    cvt_to_tfrecords(output_path=os.path.join(output_dir, 'icdar2015_test.tfrecord'),
                     data_path=test_data_dir,
                     gt_path=test_gt_dir)
