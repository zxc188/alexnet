import tensorflow as tf
from  read_data import read_data
import my_layers as layer

#---------------------------------------训练结束之后的预测---------------------------------------------------------------------------------
def interfaced(X):
    return tf.argmax(tf.nn.softmax(logits=X),1)

#---------------------------------------误差计算---------------------------------------------------------------------------------
def loss(X,Y):
    loss=tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=Y,logits=X))
    tf.summary.scalar("softmax", loss)
    return loss

#---------------------------------------训练减少误差---------------------------------------------------------------------------------
def train(loss_value, learning_rate):
    return tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss_value)


if __name__=='__main__':
    # 输入占位符
    with tf.name_scope("input"):
        x_input = tf.placeholder(tf.float32, [None, 250, 250, 3], name="x_input")
        y_input = tf.placeholder(tf.int64, [None], name="x_input")

        with tf.name_scope("placeholder"):
            image_batch, lable_batch, batch_size = read_data('./tfcode/test/*.tfrecord')

    with tf.name_scope("cnnlayer1"):
        layer1=layer.cnnlayer(x_input,64,4,11,padding='VALID')
        # with tf.name_scope("lrnlayer"):
        #     normal1=tf.nn.lrn(layer1,4,bias=1.0,alpha=0.001/9,beta=0.75,name="lrn1")
        with tf.name_scope("maxpoll1"):
            max_poll1=layer.pool(layer1)
    with tf.name_scope("cnnlayer2"):
        layer2=layer.cnnlayer(max_poll1,192,2,5)
        # with tf.name_scope("lrnlayer"):
        #     normal2 = tf.nn.lrn(layer2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name="lrn1")
        with tf.name_scope("maxpoll2"):
            max_poll2 = layer.pool(layer2)
    with tf.name_scope("cnnlayer3"):
        layer3=layer.cnnlayer(max_poll2,384,1,3)
    with tf.name_scope("cnnlayer4"):
        layer4=layer.cnnlayer(layer3,256,1,3)
    with tf.name_scope("cnnlayer5"):
        layer5=layer.cnnlayer(layer4,256,1,3)
        with tf.name_scope("maxpoll2"):
            max_poll3 = layer.pool(layer5)
    with tf.name_scope("reshape"):
        flattened = layer.resahpe(max_poll3)

    with tf.name_scope("fullylayer1"):
        fullylayer1=layer.fullylayer(flattened,4096,1024,action=tf.nn.relu)

    with tf.name_scope("fullylayer2"):
        fullylayer2=layer.fullylayer(fullylayer1,1024,1024,action=tf.nn.relu)

    with tf.name_scope("fullylayer3"):
        fullylayer3= layer.fullylayer(fullylayer2, 1024, 120,zeros=1)

    with tf.name_scope("output"):
        logits=fullylayer3
        result=tf.argmax(tf.nn.softmax(logits),1,name="result")

    # with tf.name_scope("learning_rate"):
    #     batch = tf.Variable(0, trainable=False)
    #     learning_rate = tf.train.exponential_decay(
    #             0.1, batch * 64, 25600, 0.5, staircase=True)
    #     tf.summary.scalar(name="learning_rate", tensor=learning_rate)

    with tf.name_scope("loss"):
        loss = loss(logits, y_input)
    with tf.name_scope("train"):
        train = train(loss, 0.05)
    with tf.name_scope("accuracy"):
        accuracy=tf.reduce_mean(tf.cast(tf.equal(interfaced(logits),y_input),tf.float32))
        tf.summary.scalar(name="accuracy", tensor=accuracy)


    summ = tf.summary.merge_all()
    writer=tf.summary.FileWriter("./log")
    saver=tf.train.Saver()

    sess = tf.InteractiveSession()
    # # 必须同时有全局变量和局部变量的初始化，不然会报错：
    # # OutOfRangeError (see above for traceback): RandomShuffleQueue '_134_shuffle_batch_8/random_shuffle_queue' is closed and has insufficient elements (requested 3, current size 0)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    writer.add_graph(graph=sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(logits.get_shape())
# -------------------------------------------------------------------开始训练-----------------------------------------------------------------------------------------
    for i in range(2000):
        input_image = sess.run(image_batch)
        input_lable = sess.run(lable_batch)
        # print(input_lable)
        l, t, summs, acc, losses = sess.run([logits, train, summ, accuracy, loss],
                                                feed_dict={x_input: input_image, y_input: input_lable})
        writer.add_summary(summary=summs, global_step=i)
        if i % 10 == 0:
            print('step: %d, loss: %.4f, acc: %.4f' % (i, losses, acc))
    saver.save(sess, "./model/this_model/model", global_step=i)

# ------------------------------------------------------------------开始测试--------------------------------------------------------------------------------------------
    coord.request_stop()
    coord.join(threads=threads)
    sess.close()