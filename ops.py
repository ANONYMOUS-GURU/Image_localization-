import tensorflow as tf 

def conv(X,filter_size,out_channels,stride_size=1,padding='SAME',a=None):
    X_shape=X.get_shape().as_list()
    conv_weights=tf.get_variable(name='weights',dtype=tf.float32,shape=[filter_size[0],filter_size[1],X_shape[-1],out_channels],initializer=tf.contrib.layers.xavier_initializer())
    conv_biases=tf.get_variable(name='bias',dtype=tf.float32,shape=[out_channels],initializer=tf.contrib.layers.xavier_initializer())
    conv_layer=tf.nn.conv2d(X,filter=conv_weights,strides=[1,stride_size,stride_size,1],padding=padding)
    conv_layer=tf.nn.bias_add(conv_layer,conv_biases)
    if a:
        conv_layer=a(conv_layer)
    return conv_layer

def fc(inputs, output_size,a=None):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) == 4:
        fc_weights = tf.get_variable(name='weights',shape=[input_shape[1] * input_shape[2] * input_shape[3], output_size],
            initializer=tf.truncated_normal_initializer,dtype=tf.float32)
        inputs = tf.reshape(inputs, [-1, fc_weights.get_shape().as_list()[0]])
    else:
        fc_weights = tf.get_variable(name='weights',shape=[input_shape[-1], output_size],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)

    fc_biases = tf.get_variable(name='biases',shape=[output_size],initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    fc_layer = tf.matmul(inputs, fc_weights)
    fc_layer = tf.nn.bias_add(fc_layer, fc_biases)
    if a:
        fc_layer = a(fc_layer)
    return fc_layer

def lrn(X, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0):
    return tf.nn.local_response_normalization(X, depth_radius=depth_radius, alpha=alpha, beta=beta, bias=bias)


def batch_norm(X,training=False,momentum=0.99,epsilon=0.001,name='batch_norm'):
    return tf.layers.batch_normalization(X,momentum=momentum,epsilon=epsilon,training=training,name=name)

def maxpool(X,filter_size=[3,3],stride_size=2,padding='VALID'):
    return tf.nn.max_pool(X,ksize=[1,filter_size[0],filter_size[1],1],strides=[1,stride_size,stride_size,1],padding=padding,name='MaxPool')

def avgpool(X,filter_size=3,stride_size=2,padding='VALID'):
    return tf.nn.avg_pool(X,ksize=[1,filter_size,filter_size,1],strides=[1,stride_size,stride_size,1],padding=padding,name='AvgPool')