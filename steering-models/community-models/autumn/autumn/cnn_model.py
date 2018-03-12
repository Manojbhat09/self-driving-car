import tensorflow as tf

def conv_model(features, labels, is_training, batch_norm=True, l2_reg=0.0001, dropout_rate=0.2):
   
    x_image = tf.reshape(features['image'], [-1,200,66,3])
    y_true = tf.reshape(labels['label'], [-1,1])

    # conv layer 1
    conv1 = tf.layers.conv2d(x_image, filters=24, kernel_size=5, strides=2, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
    if batch_norm:
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)

    # conv layer 2
    conv2 = tf.layers.conv2d(conv1, filters=36, kernel_size=5, strides=2, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

    # conv layer 3
    conv3 = tf.layers.conv2d(conv2, filters=48, kernel_size=5, strides=2, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
    if batch_norm:
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)

    # conv layer 4
    conv4 = tf.layers.conv2d(conv3, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

    # conv layer 5
    conv5 = tf.layers.conv2d(conv4, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
    if batch_norm:
        conv5 = tf.layers.batch_normalization(conv5, training=is_training)

    # flatten
    conv5_flat = tf.layers.flatten(conv5)

    # fully connected layer 1
    fc1 = tf.layers.dense(conv5_flat, units=1164,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
    if batch_norm:
        fc1 = tf.layers.batch_normalization(fc1, training=is_training)
    fc1 = tf.layers.dropout(fc1, rate=dropout_rate, training=is_training)

    # fully connected layer 2
    fc2 = tf.layers.dense(fc1, units=100,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
    if batch_norm:
        fc2 = tf.layers.batch_normalization(fc2, training=is_training)
    fc2 = tf.layers.dropout(fc2, rate=dropout_rate, training=is_training)

    # fully connected layer 3
    fc3 = tf.layers.dense(fc2, units=50,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
    if batch_norm:
        fc3 = tf.layers.batch_normalization(fc3, training=is_training)
    fc3 = tf.layers.dropout(fc3, rate=dropout_rate, training=is_training)

    # fully connected layer 4
    fc4 = tf.layers.dense(fc3, units=10,kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
    if batch_norm:
        fc4 = tf.layers.batch_normalization(fc4, training=is_training)
    fc4 = tf.layers.dropout(fc4, rate=dropout_rate, training=is_training)

    # fully connected layer 5 (logit)
    y = tf.layers.dense(fc4, units=1)

    return y, y_true
