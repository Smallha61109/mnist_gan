import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime

#  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
from tensorflow.examples.tutorials.mnist import input_data

def discriminator(images, reuse_variables=None):
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
    dw1 = tf.get_variable(name='dw1', shape=[5, 5, 1, 32],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    db1 = tf.get_variable(name='db1', shape=[32],
        initializer=tf.constant_initializer(0.0))
    d1 = tf.nn.conv2d(input=images, filter=dw1, strides=[1, 1, 1, 1], padding='SAME')
    d1 = d1 + db1
    d1 = tf.nn.relu(d1)
    d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    dw2 = tf.get_variable(name='dw2', shape=[5, 5, 32, 64],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    db2 = tf.get_variable(name='db2', shape=[64],
        initializer=tf.constant_initializer(0.0))
    d2 = tf.nn.conv2d(input=d1, filter=dw2, strides=[1, 1, 1, 1], padding='SAME')
    d2 = d2 + db2
    d2 = tf.nn.relu(d2)
    d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    dw3 = tf.get_variable(name='dw3', shape=[7*7*64, 1024],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    db3 = tf.get_variable(name='db3', shape=[1024],
        initializer=tf.constant_initializer(0.0))
    d3 = tf.reshape(d2, [-1, 7*7*64])
    d3 = tf.matmul(d3, dw3) + db3
    d3 = tf.nn.relu(d3)


    dw4 = tf.get_variable(name='dw4', shape=[1024, 1],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    db4 = tf.get_variable(name='db4', shape=[1],
        initializer=tf.constant_initializer(0.0))
    d4 = tf.matmul(d3, dw4) + db4

    return d4

def generator(z, batch_size, z_dim):
  #  with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope2:
    gw1 = tf.get_variable(name='gw1', shape=[z_dim, 3136], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    gb1 = tf.get_variable(name='gb1', shape=[3136],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, gw1) + gb1
    g1 = tf.reshape(g1, [-1, 56, 56, 1])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.relu(g1)


    gw2 = tf.get_variable(name='gw2', shape=[3, 3, 1, z_dim/2], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    gb2 = tf.get_variable(name='gb2', shape=[z_dim/2],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, gw2, strides=[1, 2, 2 ,1], padding='SAME')
    g2 = g2 + gb2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [56, 56])


    gw3 = tf.get_variable(name='gw3', shape=[3, 3, z_dim/2, z_dim/4], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    gb3 = tf.get_variable(name='gb3', shape=[z_dim/4],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, gw3, strides=[1, 2, 2 ,1], padding='SAME')
    g3 = g3 + gb3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [56, 56])

    gw4 = tf.get_variable(name='gw4', shape=[3, 3, z_dim/4, 1], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    gb4 = tf.get_variable(name='gb4', shape=[1],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, gw4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + gb4
    g4 = tf.sigmoid(g4)

    return g4

def main():
  mnist = input_data.read_data_sets("MNIST_data/")
  z_dimensions = 100
  batch_size = 64
  sess = tf.Session()
  #  input of gengerator
  z_placeholder = tf.placeholder(name='z_placeholder', shape=[None, z_dimensions], dtype=tf.float32)
  #  inpurt of discriminator (real image)
  x_placeholder = tf.placeholder(name='x_placeholder', shape=[None, 28, 28, 1], dtype=tf.float32)

  generated = generator(z_placeholder, batch_size, z_dimensions)

  dx = discriminator(x_placeholder)
  dg = discriminator(generated, reuse_variables=True)

  d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=dx, labels=tf.ones_like(dx)))
  d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=dg, labels=tf.zeros_like(dg)))
  g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=dg, labels=tf.ones_like(dg)))

  t_vars = tf.trainable_variables()
  d_vars = [var for var in t_vars if 'd' in var.name]
  g_vars = [var for var in t_vars if 'g' in var.name]

  d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
  d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)
  g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

  tf.get_variable_scope().reuse_variables()
  tf.summary.scalar('Generator_loss', g_loss)
  tf.summary.scalar('Discriminator_loss_real', d_loss_real)
  tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)
  tb_image = generator(z_placeholder, batch_size, z_dimensions)
  tf.summary.image('Generated_image', tb_image, 5)
  merge_summary = tf.summary.merge_all()
  logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
  #  logdir = "tensorboard/" + DateTime.DateTime.now().str + "/"
  writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

  saver = tf.train.Saver()

  sess.run(tf.global_variables_initializer())

  for i in range(300):
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    _, _, dLossReal, dLossFake = sess.run(
        [d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
        feed_dict={x_placeholder: real_image_batch, z_placeholder: z_batch})
    if i % 100 == 0:
      print("dLossReal: %f, dLossFake: %f" % (dLossReal, dLossFake))

  for i in range(100000):
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])

    _, _, dLossReal, dLossFake = sess.run(
        [d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
        feed_dict={x_placeholder: real_image_batch, z_placeholder: z_batch})

    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    _, gLoss = sess.run(
        [g_trainer, g_loss], feed_dict={x_placeholder: real_image_batch, z_placeholder: z_batch})

    if i % 100 == 0:
      print("%d: dLossReal: %f, dLossFake: %f, gLoss: %f" % (i+1, dLossReal, dLossFake, gLoss))
    if i % 10 == 0:
      #  z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
      summary = sess.run(merge_summary, feed_dict={
          x_placeholder: real_image_batch, z_placeholder: z_batch})
      writer.add_summary(summary, global_step=i)


  #  with tf.Session() as sess:
  #    sess.run(tf.global_variables_initializer())
  #    generated_image = sess.run(generated, feed_dict={z_placeholder: z_batch})
  #    generated_image = generated_image.reshape([28, 28])
  #    plt.imshow(generated_image, cmap='Greys')
  #    plt.show()


if __name__ == '__main__':
  main()
