import tensorflow as tf
import yaml
import os
from tqdm import tqdm
from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('input_yaml_path', '', 'Path to input annotations yaml file')
flags.DEFINE_string('output_record_path', '', 'Path to output TFRecord')
flags.DEFINE_integer('img_height', 0, 'Height of input images')
flags.DEFINE_integer('img_width', 0, 'Width of input images')
FLAGS = flags.FLAGS

LABEL_DICT =  {
    "Green" : 1,
    "Red" : 2,
    "Yellow" : 3,
    "off" : 4,
    }

def create_tf_example(example):
    height = FLAGS.img_height
    width = FLAGS.img_width

    filename = example['filename'] # Filename of the image. Empty if image is not from file
    filename = filename.encode()

    with tf.gfile.GFile(example['filename'], 'rb') as fid:
        encoded_image = fid.read()

    image_format = 'jpg'.encode() 

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for box in example['annotations']:
        #print("adding box")
        xmins.append(float(box['xmin'] / width))
        xmaxs.append(float((box['xmin'] + box['x_width']) / width))
        ymins.append(float(box['ymin'] / height))
        ymaxs.append(float((box['ymin']+ box['y_height']) / height))
        classes_text.append(box['class'].encode())
        classes.append(int(LABEL_DICT[box['class']]))


    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_record_path)
    examples = yaml.load(open(FLAGS.input_yaml_path, 'rb').read())

    for i in range(len(examples)):
        examples[i]['filename'] = os.path.abspath(os.path.join(os.path.dirname(FLAGS.input_yaml_path), examples[i]['filename']))
    
    counter = 0
    for example in tqdm(examples):
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())

    writer.close()



if __name__ == '__main__':
    tf.app.run()
