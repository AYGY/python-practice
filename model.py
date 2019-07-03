import tensorflow as tf
import tensorflow.contrib.slim as slim


# Inception-resnet-A
def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            # 35 x 35 x 32
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
            # weights = tf.get_variable('weight',
            #                           shape=[1, 1, 3, 32],
            #                           dtype=tf.float32,
            #                           initializer=tf.truncated_normal_initializer(stddev=0.1))
            # #正则化w
            # biases = tf.get_variable('biases',
            #                          shape=[32],
            #                          dtype=tf.float32,
            #                          initializer=tf.constant_initializer(0.1))
            # conv = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding='SAME')
            # pre_activation = tf.nn.bias_add(conv, biases)
            # conv = tf.nn.relu(pre_activation, scope='Conv2d_1x1')
            # #正则化net
        with tf.variable_scope('Branch_1'):
            # 35 x 35 x 32
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            # 35 x 35 x 32
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
            # weights_0 = tf.get_variable('weight',
            #                           shape=[1, 1, 3, 32],
            #                           dtype=tf.float32,
            #                           initializer=tf.truncated_normal_initializer(stddev=0.1))
            # # 正则化w
            # biases_0 = tf.get_variable('biases',
            #                          shape=[32],
            #                          dtype=tf.float32,
            #                          initializer=tf.constant_initializer(0.1))
            # conv1_0 = tf.nn.conv2d(net, weights_0, strides=[1, 1, 1, 1], padding='SAME')
            # pre_activation1_0 = tf.nn.bias_add(conv1_0, biases_0)
            # conv1_0 = tf.nn.relu(pre_activation1_0, scope='Conv2d_0a_1x1')
            # # 正则化net
            # weights_1 = tf.get_variable('weight',
            #                           shape=[3, 3, 3, 32],
            #                           dtype=tf.float32,
            #                           initializer=tf.truncated_normal_initializer(stddev=0.1))
            # # 正则化w
            # biases_1 = tf.get_variable('biases',
            #                          shape=[32],
            #                          dtype=tf.float32,
            #                          initializer=tf.constant_initializer(0.1))
            # conv1_1 = tf.nn.conv2d(conv1_0, weights_1, strides=[1, 1, 1, 1], padding='SAME')
            # pre_activation1_1 = tf.nn.bias_add(conv1_1, biases_1)
            # conv1_1 = tf.nn.relu(pre_activation1_1, scope='Conv2d_0b_3x3')
            # # 正则化net
        with tf.variable_scope('Branch_2'):
            # 35 x 35 x 32
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            # 35 x 35 x 32
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 32, 3, scope='Conv2d_0b_3x3')
            # 35 x 35 x 32
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
            # weights_0 = tf.get_variable('weight',
            #                           shape=[1, 1, 3, 32],
            #                           dtype=tf.float32,
            #                           initializer=tf.truncated_normal_initializer(stddev=0.1))
            # # 正则化w
            # biases_0 = tf.get_variable('biases',
            #                          shape=[32],
            #                          dtype=tf.float32,
            #                          initializer=tf.constant_initializer(0.1))
            # conv2_0 = tf.nn.conv2d(net, weights_0, strides=[1, 1, 1, 1], padding='SAME')
            # pre_activation2_0 = tf.nn.bias_add(conv2_0, biases_0)
            # conv2_0 = tf.nn.relu(pre_activation2_0, scope='Conv2d_0a_1x1')
            # # 正则化net
            # weights_1 = tf.get_variable('weight',
            #                           shape=[3, 3, 3, 32],
            #                           dtype=tf.float32,
            #                           initializer=tf.truncated_normal_initializer(stddev=0.1))
            # # 正则化w
            # biases_1 = tf.get_variable('biases',
            #                          shape=[32],
            #                          dtype=tf.float32,
            #                          initializer=tf.constant_initializer(0.1))
            # conv2_1 = tf.nn.conv2d(conv2_0, weights_1, strides=[1, 1, 1, 1], padding='SAME')
            # pre_activation2_1 = tf.nn.bias_add(conv2_1, biases_1)
            # conv2_1 = tf.nn.relu(pre_activation2_1, scope='Conv2d_0b_3x3')
            # # 正则化net
            # weights_2 = tf.get_variable('weight',
            #                             shape=[3, 3, 3, 32],
            #                             dtype=tf.float32,
            #                             initializer=tf.truncated_normal_initializer(stddev=0.1))
            # # 正则化w
            # biases_2 = tf.get_variable('biases',
            #                            shape=[32],
            #                            dtype=tf.float32,
            #                            initializer=tf.constant_initializer(0.1))
            # conv2_2 = tf.nn.conv2d(conv2_1, weights_2, strides=[1, 1, 1, 1], padding='SAME')
            # pre_activation2_2 = tf.nn.bias_add(conv2_2, biases_2)
            # conv2_2 = tf.nn.relu(pre_activation2_2, scope='Conv2d_0c_3x3')
            # # 正则化net
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        # weights_0 = tf.get_variable('weight',
        #                             shape=[1, 1, 3, net.get_shape()[3]],
        #                             dtype=tf.float32,
        #                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        # # 正则化w
        # biases_0 = tf.get_variable('biases',
        #                            shape=[net.get_shape()[3]],
        #                            dtype=tf.float32,
        #                            initializer=tf.constant_initializer(0.1))
        # 35 x 35 x 96
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        # 35 x 35 x 256
        # up = tf.nn.bias_add(conv2_0, biases_0)
        # scale = 0.17
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Reduction-A
def reduction_a(net, k, l, m, n):
    # 192, 12, 256, 384
    with tf.variable_scope('Branch_0'):
        # 17 x 17 x 384
        tower_conv = slim.conv2d(net, n, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        # 35 x 35 x 192
        tower_conv1_0 = slim.conv2d(net, k, 1, scope='Conv2d_0a_1x1')
        # 35 x 35 x 192
        tower_conv1_1 = slim.conv2d(tower_conv1_0, l, 3, scope='Conv2d_0b_3x3')
        # 17 x 17 x 256
        tower_conv1_2 = slim.conv2d(tower_conv1_1, m, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        # 17 x 17 x 256
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
    # 17 x 17 x 896
    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
    return net


# Inception-ResNet-B
def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Block17', [net], reuse):
        with tf.variable_scope('Branch_0'):
            # 17 x 17 x 128
            tower_conv = slim.conv2d(net, 128, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            # 17 x 17 x 128
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            # 17 x 17 x 128
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 128, [1, 7], scope='Conv2d_0b_1x7')
            # 17 x 17 x 128
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 128, [7, 1], scope='Conv2d_0c_7x1')
        # 17 x 17 x 256
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        # 17 x 17 x 896
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conved_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Reduction-B
def rection_b(net):
    with tf.variable_scope('Branch_0'):
        # 17 x 17 x 256
        tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        # 8 x 8 x 384
        tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        # 17 x 17 x 256
        tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        # 8 x 8 x 256
        tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        # 17 x 17 x 256
        tower_conv2_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        # 17 x 17 x 256
        tower_conv2_1 = slim.conv2d(tower_conv2_0, 256, 3, scope='Conv2d_0b_3x3')
        # 8 x 8 x 256
        tower_conv2_2 = slim.conv2d(tower_conv2_1, 256, 3, stride=2, padding='VALID', scope='Conv2d_0c_3x3')
    with tf.variable_scope('Branch_3'):
        # 8 x 8 x 896
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
    # 8 x 8 x 1792
    net = tf.concat([tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)
    return net


# Inception-ResNet-C
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            # 8 x 8 x 192
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            # 8 x 8 x 192
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            # 8 x 8 x 192
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 192, [1, 3], scope='Conv2d_0b_1x3')
            # 8 x 8 x 192
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [3, 1], scope='Conv2d_0c_3x1')
        # 8 x 8 x 384
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        # 8 x 8 x 1792
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        # scale = 0.2
        net += scale * up
        if activation_fn:
            activation_fn(net)
    return net


# Inception-ResNet-V1
def inception_resnet_v1(inputs, is_training=True, dropout_keep_prob=0.8,
                        bottleneck_layer_size=0.8, reuse=None, scope='InceptionResNetV1'):
    end_points = {}
    with tf.variable_scope(scope, 'InceptionResNetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):

                # 149 x 149 x 32
                net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net
                # print("vvv")
                # 147 x 147 x 32
                net = slim.conv2d(net, 32, 3, padding='VALID', scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net
                # 147 x 147 x 64
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                end_points['Conv2d_2b_3x3'] = net
                # 73 x 73 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_3a_3x3')
                end_points['MaxPoole_3a_3x3'] = net
                # 73 x 73 x 80
                net = slim.conv2d(net, 80, 1, padding='VALID', scope='Conv2d_3b_1x1')
                end_points['Conv2d_3b_1x1'] = net
                # 71 x 71 x 192
                net = slim.conv2d(net, 192, 3, padding='VALID', scope='Conv2d_4a_3x3')
                end_points['Conv2d_4a_3x3'] = net
                # 35 x 35 x 256
                net = slim.conv2d(net, 256, 3, stride=2, padding='VALID', scope='Conv2d_4b_3x3')
                end_points['Conv2d_4b_3x3'] = net
                # print("dddd")

                # 5 x Inception-resnet-A
                net = slim.repeat(net, 5, block35, scale=0.17)
                end_points['Mixed_5a'] = net

                # Reduction-A
                with tf.variable_scope('Mixed_6a'):
                    net = reduction_a(net, 192, 192, 256, 384)
                end_points['Mixed_6a'] = net

                # 10 x Inception-Resnet-B
                net = slim.repeat(net, 10, block17, scale=0.10)
                end_points['Mixed_6b'] = net
                # Reduction-B
                with tf.variable_scope('Mixed_7a'):
                    net = rection_b(net)
                end_points['Mixed_7a'] = net

                # 5 x Inception-Resnet-C
                net = slim.repeat(net, 5, block8, scale=0.20)
                end_points['Mixed_8a'] = net
                net = block8(net, activation_fn=None)
                end_points['Mixed_8b'] = net

                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a_3x3')
                    net = slim.flatten(net)
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='Dropout')
                    end_points['PreLogitsFlatten'] = net
                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope='Bottleneck', reuse=False)
    return net, end_points


def inference(images, keep_probability, phase_train=True,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        'decay': 0.995,
        'epsilon': 0.001,
        'updates_collections': None,
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        # print("aaaaa")
        return inception_resnet_v1(images, is_training=phase_train, dropout_keep_prob=keep_probability,
                                   bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)


# def inference(images,batch_size,n_classes):
#     with tf.variable_scope('conv1') as scope:
#         weights = tf.get_variable('weight',
#                                   shape = [3,3,3,16],
#                                   dtype = tf.float32,
#                                   initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
#         biases = tf.get_variable('biases',
#                                  shape = [16],
#                                  dtype = tf.float32,
#                                  initializer = tf.constant_initializer(0.1))
#         conv = tf.nn.conv2d(images,weights,strides = [1,1,1,1],padding = 'SAME')
#         pre_activation = tf.nn.bias_add(conv,biases)
#         conv1 = tf.nn.relu(pre_activation,name = scope.name)
#
#
#
#     with tf.variable_scope('pooling_lrn') as scope:
#         pool1 = tf.nn.max_pool(conv1,ksize = [1,3,3,1],strides = [1,2,2,1],
#                                padding = 'SAME',name = 'pooling1')
#         norm1 = tf.nn.lrn(pool1,depth_radius = 4,bias = 1.0,alpha = 0.001/9.0,
#                           bata = 0.75,name = 'norm1')
#
#
#     with tf.variable_scope('conv2') as scope:
#         weights = tf.get_variable('weight',
#                                   shape = [3,3,3,16],
#                                   dtype = tf.float32,
#                                   initializer = tf.truncated_normal_initializer(stddev = 0.1,dtype = tf.float32))
#         biases = tf.get_variable('biases',
#                                   shape = [16],
#                                  dtype = tf.float32,
#                                  initializer = tf.constant_initializer(0.1))
#         conv = tf.nn.conv2d(norm1,weights,strides = [1,1,1,1],padding = 'SAME')
#         pre_activation = tf.nn.bias_add(conv,biases)
#         conv2 = tf.nn.relu(pre_activation,name = 'conv2')
#
#
#
#     with tf.variable_scope('pooling2_lrn') as scope:
#         norm2 = tf.nn.lrn(conv2,depth_radius = 4,bias = 1.0,alpha = 0.001/9.0,
#                           bata = 0.75,name = 'norm2')
#         pool2 = tf.nn.max_pool(norm2,ksize = [1,3,3,1],strides = [1,1,1,1],
#                                padding = 'SAME',name = 'pooling2')
#
#     with tf.variable_scope('local3') as scope:
#         reshape = tf.reshape(pool2,shape = [batch_size,-1])
#         dim = reshape.get_shape()[1].value
#         weights = tf.get_variable('weights',
#                                   shape=[dim,128],
#                                   dtype=tf.float32,
#                                   initializer = tf.constant_initializer(stddev=0.005,dtype=tf.flfloat32))
#         biases = tf.get_variable('biases',
#                                  shape=[128],
#                                  dtype = tf.float32,
#                                  initializer = tf.constant_initializer(0.1))
#         local3 = tf.nn.relu(tf.matmul(reshape,weights)+biases,name = scope.name)
#
#     with tf.variable_scope('local4') as scope:
#         weights = tf.get_variable('weights',
#                                   shape=[128,128],
#                                   dtype=tf.float32,
#                                   initializer = tf.constant_initializer(stddev=0.005,dtype=tf.flfloat32))
#         biases = tf.get_variable('biases',
#                                  shape=[128],
#                                  dtype = tf.float32,
#                                  initializer = tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
#         local4 = tf.nn.relu(tf.matmul(local3,weights)+biases,name = 'local4')
#
#
#     with tf.variable_scope('softmax_linear') as scope:
#         weights = tf.get_variable('softmax_linear',
#                                   shape=[128,n_classes],
#                                   dtype=tf.float32,
#                                   initializer = tf.constant_initializer(stddev=0.005,dtype=tf.float32))
#         biases = tf.get_variable('biases',
#                                  shape=[n_classes],
#                                  dtype = tf.float32,
#                                  initializer = tf.constant_initializer(0.1))
#         softmax_linear = tf.add(tf.matmul(local4,weights),biases,name = 'softmax_linear')
#     return softmax_linear
#
# def losses(logits,labels):
#     with tf.variable_scope('loss') as scope:
#         cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
#                         (logits=logits,labels = labels,name='xentropy_per_example')
#         loss = tf.reduce_mean(cross_entropy,name = 'loss')
#         tf.summary.scalar(scope.name+'/loss',loss)
#     return loss
#
# def trainning(loss,learning_rate):
#     with tf.name_scope('optimizer'):
#         optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
#         global_step = tf.Variable(0,name = 'global_step',trainable=False)
#         train_op = optimizer.minimize(loss,global_step=global_step)
#     return train_op
#
# def evaluation(logits,labels):
#     with tf.variable_scope('accuracy') as scope:
#         correct = tf.nn.in_top_k(logits,labels,1)
#         correct = tf.cast(correct,tf.flaot16)
#         accuracy = tf.reduce_mean(correct)
#         tf.summary.scalar(scope.name+'/accuracy',accuracy)
#     return accuracy


















