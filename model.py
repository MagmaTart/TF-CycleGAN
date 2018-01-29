import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

# Instance Normalization function : Implemented by Hardik Bansal
# https://github.com/hardikbansal
def instance_norm(input):
    with tf.variable_scope('instance_norm'):
        eps = 1e-5

        mean, var = tf.nn.moments(input, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [input.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [input.get_shape()[-1]],
                                 initializer=tf.constant_initializer(0.0))
        output = scale * tf.div(input - mean, tf.sqrt(var + eps)) + offset
        return output

def leaky_relu(input, slope=0.2):
    return tf.nn.relu(input) - slope * tf.nn.relu(-input)

class Model:
    def __init__(self):
        pass

    def residual_block(self, input, num_outputs, kernel_size, stride, name='res_block'):
        with tf.variable_scope(name):
            conv1 = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            conv1 = slim.conv2d(conv1, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride, padding='VALID',
                                activation_fn=tf.nn.relu, normalizer_fn=instance_norm,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
            conv2 = tf.pad(conv1, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            conv2 = slim.conv2d(conv2, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride, padding='VALID',
                                activation_fn=None, normalizer_fn=instance_norm,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02))

            output = tf.nn.relu(input + conv2)
            return output

    def generator(self, input, reuse=False, name='generator'):
        with tf.variable_scope(name, reuse=reuse):
            with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, normalizer_fn=instance_norm,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                conv = slim.conv2d(input, num_outputs=32, kernel_size=7, stride=1)
                conv = slim.conv2d(conv, num_outputs=64, kernel_size=3, stride=2, padding='SAME')
                conv = slim.conv2d(conv, num_outputs=128, kernel_size=3, stride=2, padding='SAME')

            conv = self.residual_block(input=conv, num_outputs=128, kernel_size=3, stride=1, name='rblock1')
            conv = self.residual_block(input=conv, num_outputs=128, kernel_size=3, stride=1, name='rblock2')
            conv = self.residual_block(input=conv, num_outputs=128, kernel_size=3, stride=1, name='rblock3')
            conv = self.residual_block(input=conv, num_outputs=128, kernel_size=3, stride=1, name='rblock4')
            conv = self.residual_block(input=conv, num_outputs=128, kernel_size=3, stride=1, name='rblock5')
            conv = self.residual_block(input=conv, num_outputs=128, kernel_size=3, stride=1, name='rblock6')
            conv = self.residual_block(input=conv, num_outputs=128, kernel_size=3, stride=1, name='rblock7')
            conv = self.residual_block(input=conv, num_outputs=128, kernel_size=3, stride=1, name='rblock8')
            conv = self.residual_block(input=conv, num_outputs=128, kernel_size=3, stride=1, name='rblock9')

            with slim.arg_scope([slim.conv2d_transpose], activation_fn=tf.nn.relu, normalizer_fn=instance_norm,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                conv = slim.conv2d_transpose(conv, num_outputs=64, kernel_size=3, stride=2, padding='SAME')
                conv = slim.conv2d_transpose(conv, num_outputs=32, kernel_size=3, stride=2, padding='SAME')
                conv = slim.conv2d(conv, num_outputs=3, kernel_size=7, stride=1, padding='SAME')

            return conv

    def discriminator(self, input, reuse=False, name='discriminator'):
        patch = tf.random_crop(input, [1, 70, 70, 3])
        with tf.variable_scope(name, reuse=reuse):
            with slim.arg_scope([slim.conv2d], kernel_size=4, stride=2, activation_fn=leaky_relu, padding='SAME',
                                normalizer_fn=instance_norm, weights_initializer=layers.xavier_initializer()):
                conv = slim.conv2d(patch, num_outputs=64)
                conv = slim.conv2d(conv, num_outputs=128)
                conv = slim.conv2d(conv, num_outputs=256)
                conv = slim.conv2d(conv, num_outputs=512)

            output = slim.conv2d(conv, num_outputs=1, kernel_size=4, stride=1, activation_fn=leaky_relu,
                                 weights_initializer=layers.xavier_initializer())
            output = slim.flatten(output)
            return output

    def build(self):
        cyc_lambda = 10.0

        # TODO : Normalize Input image. (divide by 127 and subtract 1)
        self.domain_x_image = tf.placeholder(tf.float32, [None, 256, 256, 3])
        self.domain_y_image = tf.placeholder(tf.float32, [None, 256, 256, 3])

        # L_GAN (G, D_Y, X, Y)
        self.fake_g = self.generator(self.domain_x_image, name='G')
        self.real_dy = self.discriminator(self.domain_y_image, name='dy')
        self.fake_dy = self.discriminator(self.fake_g, name='dy', reuse=True)

        self.loss_g = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.fake_dy), self.fake_dy) * 100.0
        self.loss_dy_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(self.real_dy), self.real_dy))
        self.loss_dy_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.fake_dy), self.fake_dy))
        self.loss_dy = self.loss_dy_real + self.loss_dy_fake

        # L_GAN (F, D_X, Y, X)
        self.fake_f = self.generator(self.domain_y_image, name='F')
        self.real_dx = self.discriminator(self.domain_x_image, name='dx')
        self.fake_dx = self.discriminator(self.fake_f, name='dx', reuse=True)

        self.loss_f = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.fake_dx), self.fake_dx) * 100.0
        self.loss_dx_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(self.real_dx), self.real_dx))
        self.loss_dx_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.fake_dx), self.fake_dx))
        self.loss_dx = self.loss_dx_real + self.loss_dx_fake

        # Cycle Consistency Loss
        self.fgx = self.generator(self.generator(self.domain_x_image, name='G', reuse=True), name='F', reuse=True)
        self.gfy = self.generator(self.generator(self.domain_y_image, name='F', reuse=True), name='G', reuse=True)
        self.loss_cycle_x = tf.reduce_mean(tf.abs(self.fgx - self.domain_x_image))
        self.loss_cycle_y = tf.reduce_mean(tf.abs(self.gfy - self.domain_y_image))
        self.loss_cycle = (self.loss_cycle_x + self.loss_cycle_y) * cyc_lambda

        # Optimizers
        self.optimizer_g = tf.train.AdamOptimizer(0.0002).minimize(self.loss_g)
        self.optimizer_f = tf.train.AdamOptimizer(0.0002).minimize(self.loss_f)
        self.optimizer_dy = tf.train.AdamOptimizer(0.0002).minimize(self.loss_dy)
        self.optimizer_dx = tf.train.AdamOptimizer(0.0002).minimize(self.loss_dx)
        self.optimizer_cycle = tf.train.AdamOptimizer(0.0002).minimize(self.loss_cycle)





