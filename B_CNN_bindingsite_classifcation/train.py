import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import LeNet5_infernence
import os
import numpy as np
import h5py

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 60
MOVING_AVERAGE_DECAY = 0.99



def bindingsite_data_read():


    print("please input the hdf5 path：")
    hdf5_Path = r'F:\DNA_RNA_deeplearning\RNA_file\RNA25X25'
    os.chdir(hdf5_Path)
    print("you input hdf5 path is :", os.getcwd())
    hdf5_Path_listdir = os.listdir(hdf5_Path)

    data_x = np.zeros((1,25,26))
    data_y = np.zeros((1))

    if os.path.exists(hdf5_Path+"\\total.hdf5") :
        print("total.hdf5 exit")
        f = h5py.File('total.hdf5', 'r')
        train_x = f['train_x']
        train_y = f['train_y']
        data_x = np.append(data_x, train_x, axis=0)
        data_y = np.append(data_y, train_y, axis=0)
    else:
        print("total.hdf5 not exit")
        for hdf5_file_name in hdf5_Path_listdir:
            f = h5py.File(hdf5_file_name, 'r')
            train_x = f['train_x']
            train_y = f['train_y']
            print(train_x.shape)
            if np.sum(train_y, axis=0) != 0:
                # data = np.append(data,train, axis = 0)
                data_x = np.append(data_x,train_x, axis = 0)
                data_y = np.append(data_y,train_y, axis = 0)
        f=h5py.File('total.hdf5',"w")
        f.create_dataset('train_x', data = data_x[1:,:,:])
        f.create_dataset('train_y', data = data_y[1:])

    data_y = data_y[1:]
    data_x = data_x[1:,:,:]


    return  data_x.shape[0],data_x,data_y


batch_num = 0


# def balance_binding_site():
#     row_num = 0
#     positive_data_x = np.zeros((1,25,26))
#     positive_data_y = np.zeros((1))
#     negative_data_x = np.zeros((1,25,26))
#     negative_data_y = np.zeros((1))
#
#     print(data_x.shape)
#
#     while row_num < data_y.shape[0]:
#         if data_y[0]:
#             positive_data_x = np.append(positive_data_x, data_x[row_num:row_num+1,:,:], axis=0)
#             positive_data_y = np.append(positive_data_y, data_y[row_num:row_num+1], axis=0)
#         else:
#             negative_data_x = np.append(negative_data_x, data_x[row_num:row_num+1,:,:], axis=0)
#             negative_data_y = np.append(negative_data_y, data_y[row_num:row_num+1], axis=0)
#         row_num = row_num + 1
#
#     positive_data_x = positive_data_x[1:,:,:]
#     positive_data_y = positive_data_y[1:]
#     negative_data_x = positive_data_x[1:,:,:]
#     negative_data_y = positive_data_y[1:]
#
#     print(positive_data_x.shape)
#     print(negative_data_x.shape)
#
#     global train_data_x
#     global train_data_y
#
#     row_num = 0
#     p_row_num = 0
#     n_row_num = 0
#     while row_num < positive_data_y.shape[0] :
#         if row_num%2:
#             train_data_x= np.append(train_data_x, data_x[p_row_num,:,:], axis=0)
#             train_data_y= np.append(train_data_y, data_y[p_row_num,:,:], axis=0)
#             p_row_num = p_row_num + 1
#         else:
#             train_data_y= np.append(train_data_y, data_y[n_row_num,:,:], axis=0)
#             train_data_x= np.append(train_data_x, data_x[n_row_num,:,:], axis=0)
#             n_row_num = n_row_num + 1
#
#     print(train_data_x.shape)
from sklearn.model_selection import train_test_split

def bindingsite_data_next_batch(BATCH_SIZE):

    global batch_num
    global data_x
    global data_y

    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y,test_size=0)

    batch_train_x = X_train[batch_num:batch_num+BATCH_SIZE,:,:]
    batch_train_y = y_train[batch_num:batch_num+BATCH_SIZE]
    batch_num = batch_num + BATCH_SIZE
    return batch_train_x,batch_train_y


def train(all_site_num):
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        25,
        26,
        LeNet5_infernence.NUM_CHANNELS],
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 1], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = LeNet5_infernence.inference(x, False, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        all_site_num / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            # xs, ys = mnist.train.next_batch(BATCH_SIZE)
            xs, ys = bindingsite_data_next_batch(BATCH_SIZE)

            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                25,
                26,
                LeNet5_infernence.NUM_CHANNELS))
            reshaped_ys = np.reshape(ys, (
                BATCH_SIZE,
                1))
            # print(reshaped_xs.shape)
            # print(reshaped_ys.shape)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: reshaped_ys})

            if i % 10 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))


def main(argv=None):
    # mnist = input_data.read_data_sets(
    #     r"G:\tensorflow-tutorial-master\Deep_Learning_with_TensorFlow\datasets\MNIST_data", one_hot=True)
    global data_x
    global data_y
    all_site_num,data_x,data_y = bindingsite_data_read()
    train(all_site_num)


if __name__ == '__main__':
    main()