import tensorflow as tf
import cv2

from load_data import Loader
from model import Model

loader = Loader()
model = Model()

loader.set_paths()
loader.load_train_images()

sess = tf.Session()

model.build()

sess.run(tf.global_variables_initializer())

for i in range(10001):
    imagex, imagey = loader.get_train_batch(batch_size=4)
    feed_dict = {model.domain_x_image:imagex, model.domain_y_image:imagey}
    sess.run([model.optimizer_dy, model.optimizer_dx, model.optimizer_cycle], feed_dict=feed_dict)
    sess.run([model.optimizer_dy, model.optimizer_dx, model.optimizer_cycle], feed_dict=feed_dict)
    sess.run([model.optimizer_dy, model.optimizer_dx, model.optimizer_cycle], feed_dict=feed_dict)
    sess.run([model.optimizer_g, model.optimizer_f], feed_dict=feed_dict)
    sess.run([model.optimizer_g, model.optimizer_f], feed_dict=feed_dict)
    sess.run([model.optimizer_g, model.optimizer_f], feed_dict=feed_dict)
    # sess.run([model.optimizer_g, model.optimizer_f], feed_dict=feed_dict)
    # sess.run([model.optimizer_g, model.optimizer_f], feed_dict=feed_dict)

    losses = sess.run([model.loss_dy, model.loss_dx, model.loss_cycle, model.loss_g, model.loss_f], feed_dict=feed_dict)

    if i % 10 == 0:
        imagex, imagey = loader.get_train_batch(batch_size=1)
        test = sess.run(model.fake_g, feed_dict={model.domain_x_image:imagex})
        test = test[0]
        fgx = sess.run(model.fgx, feed_dict={model.domain_x_image:imagex})
        fgx = fgx[0]
        cv2.imwrite('./samples/' + str(i) + '0.jpg', imagex[0] * 127.5)
        cv2.imwrite('./samples/' + str(i) + '1.jpg', test * 127.5)
        cv2.imwrite('./samples/' + str(i) + '2.jpg', fgx * 127.5)

    print('%d -> Dy : %8f, Dx : %8f, Cycle : %8f, G : %8f, F : %8f' % (i, losses[0], losses[1], losses[2], losses[3], losses[4]))

