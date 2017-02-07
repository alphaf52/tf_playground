
import tensorflow as tf
import time
import math

def conv(inputs, filter_width, filter_height,
         stride, out_channels):
    total_size = filter_width * filter_height * out_channels
    with tf.device('/cpu:0'):
        kernel = tf.get_variable(
            initializer=tf.truncated_normal(
                [filter_width, filter_height, 1, out_channels],
                dtype=tf.float32, stddev=4.0 / math.sqrt(float(total_size))),
            name='weights')
        biases = tf.get_variable(
            initializer=tf.constant(
                0.0, shape=[out_channels], dtype=tf.float32),
            trainable=True, name='biases')
    outputs = tf.nn.relu(
        tf.nn.bias_add(
            tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1],
                         padding='VALID'), biases))
    print("input size:", inputs.get_shape())
    print("output size:", outputs.get_shape())
    return outputs, kernel


def run():
    filter_width = 11
    stride = 4
    out_channels = 32
    filter_height = 41

    batch_size = 128
    freq_size = 161
    length = 1000

    inputs = tf.get_variable(
        initializer=tf.truncated_normal(
            [batch_size, length, freq_size, 1],
            dtype=tf.float32, stddev=4.0 / math.sqrt(float(100))),
        name='input')

    with tf.variable_scope('conv_1'):
        conv_outputs_1, kernel_1 = conv(
                inputs=inputs, filter_width=filter_width,
                filter_height=filter_height,
                stride=stride, out_channels=out_channels)

    length_1 = (length - filter_width) / stride + 1
    print("length_1:", length_1)
    conv_outputs_1 = tf.reshape(conv_outputs_1, [batch_size, length_1, -1, 1])

    with tf.variable_scope('conv_2'):
        conv_outputs_2, kernel_2 = conv(
                inputs=conv_outputs_1, filter_width=filter_width,
                filter_height=filter_height,
                stride=stride, out_channels=out_channels)

    cost = tf.reduce_sum(conv_outputs_2)
    inputs_grad = tf.gradients(cost, [inputs])[0]
    inter_grad = tf.gradients(cost, [conv_outputs_1])[0]
    kernel_grad_1 = tf.gradients(cost, [kernel_1])[0]
    kernel_grad_2 = tf.gradients(cost, [kernel_2])[0]

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        start_time = time.time()
        o = session.run([cost])
        print("forward passed time:", time.time() - start_time)

        start_time = time.time()
        g = session.run([kernel_grad_2])
        print("kernel_grad_2 passed time:", time.time() - start_time)

        start_time = time.time()
        g = session.run([inter_grad])
        print("inter_grad passed time:", time.time() - start_time)

        start_time = time.time()
        g = session.run([kernel_grad_1])
        print("kernel_grad_1 passed time:", time.time() - start_time)

        start_time = time.time()
        g = session.run([inputs_grad])
        print("inputs_grad passed time:", time.time() - start_time)


if __name__ == "__main__":
    run()

