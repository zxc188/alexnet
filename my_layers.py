import tensorflow as tf
from tensorflow.python.ops import init_ops

def weights_initializer_random_normal(shape,stddev):
    return init_ops.truncated_normal_initializer(shape, stddev)

#---------------------------------------卷积层---------------------------------------------------------------------------------
def cnnlayer(input,output_size,stride,ksize,padding='SAME'):
    #W=tf.Variable(tf.truncated_normal([5,5,input_size,output_size],stddev=0.1),name="W")
    #b = tf.Variable(tf.zeros(shape=[output_size],dtype=tf.float32), name="b")
    #conver2=tf.nn.conv2d(input,W,strides=[1,one,two,1],padding='SAME')
    #conver2_layer=tf.nn.relu(conver2+b)
    conver2_layer = tf.contrib.layers.convolution2d(
        input,
        # weights_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
        # biases_initializer=tf.constant_initializer(0.1),
        normalizer_fn=tf.contrib.layers.batch_norm,
        num_outputs=output_size,
        kernel_size=(ksize, ksize),
        activation_fn=tf.nn.relu,
        stride=(stride, stride),
        padding=padding
    )
    # tf.summary.histogram(name="W",values=W)
    # tf.summary.histogram(name="b",values=b)
    return conver2_layer
#---------------------------------------池化层---------------------------------------------------------------------------------
def pool(input):
    return tf.contrib.layers.max_pool2d(
        inputs=input,
        kernel_size=[3,3],
        padding='VALID')

#---------------------------------------全连接层---------------------------------------------------------------------------------
def fullylayer(input,input_size,output_size,zeros=0,action=None,drop=0):
    W=tf.Variable(tf.truncated_normal([input_size,output_size],stddev=0.1),name="W")
    hidden_layer_three=tf.matmul(input,W)
    if zeros==1:
        b = tf.Variable(tf.zeros(shape=[output_size], dtype=tf.float32), name="b")
        hidden_layer_three=tf.nn.bias_add(hidden_layer_three,b)
    else:
        b = tf.constant(0.1,name='b')
        hidden_layer_three=hidden_layer_three+b

    if action!=None:
        hidden_layer_three=tf.nn.relu(hidden_layer_three)

    if drop!=0:
        hidden_layer_three = tf.contrib.layers.dropout(hidden_layer_three, drop, is_training=True)
    tf.summary.histogram('W',W)
    tf.summary.histogram('b',b)
    return hidden_layer_three

#---------------------------------------更改shape---------------------------------------------------------------------------------
def resahpe(input):
    return tf.contrib.layers.flatten(input)