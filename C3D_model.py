import tensorflow as tf
import tensorflow.contrib.slim as slim

def C3D(input, num_classes, keep_pro=0.5):
    with tf.variable_scope('C3D'):
        with slim.arg_scope([slim.conv3d],
                            padding='SAME',
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            activation_fn=tf.nn.relu,
                            kernel_size=[3, 3, 3],
                            stride=[1, 1, 1]
                            ):
            net = slim.conv3d(input, 64, scope='conv1')
            net = slim.max_pool3d(net, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding='SAME', scope='max_pool1')
            net = slim.conv3d(net, 128, scope='conv2')
            net = slim.max_pool3d(net, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding='SAME', scope='max_pool2')
            net = slim.repeat(net, 2, slim.conv3d, 256, scope='conv3')
            net = slim.max_pool3d(net, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding='SAME', scope='max_pool3')
            net = slim.repeat(net, 2, slim.conv3d, 512, scope='conv4')
            net = slim.max_pool3d(net, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding='SAME', scope='max_pool4')
            net = slim.repeat(net, 2, slim.conv3d, 512, scope='conv5')
            net = slim.max_pool3d(net, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding='SAME', scope='max_pool5')

            net = tf.reshape(net, [-1, 512 * 4 * 4])
            net = slim.fully_connected(net, 4096, weights_regularizer=slim.l2_regularizer(0.0005), scope='fc6')
            net = slim.dropout(net, keep_pro, scope='dropout1')
            net = slim.fully_connected(net, 4096, weights_regularizer=slim.l2_regularizer(0.0005), scope='fc7')
            net = slim.dropout(net, keep_pro, scope='dropout2')
            out = slim.fully_connected(net, num_classes, weights_regularizer=slim.l2_regularizer(0.0005), \
                                       activation_fn=None, scope='out')

            return out


