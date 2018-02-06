import tensorflow as tf
import glob


#---------------------------------------图像数据的归一化---------------------------------------------------------------------------------
def input_float(iamge_beach):
    return tf.image.convert_image_dtype(iamge_beach,tf.float32)

#---------------------------------------将标签对应到  0-119  ---------------------------------------------------------------------------------
def lables(lable_batch):
    # 得到标签列表
    lablees = list(map(lambda c: c.split("\\")[1], glob.glob(("./image_dog\\*"))))
    # 将标签变成一个以为的张量
    train_lable = tf.map_fn(lambda l: tf.where(tf.equal(lablees, l))[0, 0:1][0], lable_batch, dtype=tf.int64)
    return train_lable

#---------------------------------------读取数据---------------------------------------------------------------------------------
def read_data(file_path):
    filename_output_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(["./tfcode/train/*.tfrecord"]))  # 读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_output_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'image': tf.FixedLenFeature([], tf.string)
                                       })  # 取出包含image和label的feature对象
    recode_image = tf.decode_raw(features['image'], tf.uint8)
    real_image = tf.reshape(recode_image, shape=[250, 250, 3])
    lable = tf.cast(features['label'], tf.string)
    min_after_dequeue = 5000
    batch_size = 60
    capacity =min_after_dequeue+10*batch_size
    image_batch, lable_batch = tf.train.shuffle_batch([real_image, lable],
                                              batch_size=batch_size,
                                              capacity=capacity,
                                              min_after_dequeue=min_after_dequeue)
    image_batch=input_float(image_batch)
    lable_batch=lables(lable_batch)
    return  image_batch, lable_batch,batch_size



