import tensorflow as tf
import glob
from itertools import groupby
from collections import defaultdict


sess = tf.InteractiveSession()

#查找符合一定规则的所有文件，并将文件名以lis形式返回。
image_filenames_0 = glob.glob("./image_dog/n02*/*.jpg")

#这句是我添加的。因为读到的路径形式为：'./imagenet-dogs\\n02085620-Chihuahua\\n02085620_10074.jpg'，路径分隔符中除第1个之外，都是2个反斜杠，与例程不一致。这里将2个反斜杠替换为斜杠
image_filenames = list(
    map(lambda image: image.replace('\\', '/'), image_filenames_0))

image_filename_with_breed = map(
    lambda filename: (filename.split("/")[2], filename), image_filenames)

#用list类型初始化training和testing数据集，用defaultdict的好处是为字典中不存在的键提供默认值
training_dataset = defaultdict(list)
testing_dataset = defaultdict(list)

for dog_breed, breed_images in groupby(image_filename_with_breed,
                                       lambda x: x[0]):

    #enumerate的作用是列举breed_images中的所有元素，可同时返回索引和元素，i和breed_image
    #的最后一个值分别是：168、('n02116738-African_hunting_dog', './imagenet-dogs/
    #n02116738-African_hunting_dog/n02116738_9924.jpg')
    for i, breed_image in enumerate(breed_images):

        #因为breed_images是按类分别存储的，所以下面是将大约20%的数据作为测试集，大约80%的
        #数据作为训练集。
        #testing_dataset和training_dataset是两个字典，testing_dataset中
        #的第一个元素是 'n02085620-Chihuahua': ['./imagenet-dogs/n02085620-Chihuahua/
        #n02085620_10074.jpg', './imagenet-dogs/n02085620-Chihuahua/
        #n02085620_11140.jpg',.....]
        print(breed_image)
        if i % 5 == 0:
            testing_dataset[dog_breed].append(breed_image[1])
        else:
            training_dataset[dog_breed].append(breed_image[1])

def write_tfrecord(data,location,sess):
    writer = None
    index = 0
    for whichs,filenames in data.items():
        for filename in filenames:
            if index%100==0:
                if writer:
                    writer.close()
                record_name="{location}-{index}.tfrecord".format(location=location,index=index)
                writer=tf.python_io.TFRecordWriter(record_name)
            index=index+1
            image=tf.read_file(filename)
            try:
                image_dec=tf.image.decode_jpeg(image)
            except:
                continue
            #调整图像的大小,不是修剪，相当于缩放
            resize_image = tf.image.resize_images(images=image_dec, size=[250, 250])
            image_bytes = sess.run(tf.cast(resize_image, tf.uint8)).tobytes()
            image_lable = whichs.encode('utf-8')
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_lable])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }))
            writer.write(example.SerializeToString())
    writer.close()

write_tfrecord(training_dataset,"./tfcode/train/training_image",sess)