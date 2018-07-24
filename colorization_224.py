import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import numpy as np

# Path of saved model
save_file = '/home/test/deeplearning/model.ckpt'

# Path of input images
file_path = '/home/test/deeplearning/MyFile.h5'

fr = h5py.File(file_path, 'r')
binary_image = np.array(fr.get('binary_img'))
color_image = np.array(fr.get('color_img'))

# Options
total_epoch = 1000
batch_size = 64
n_input = 224*224*3
learning_rate = 0.0002

# Neural Net Basic Models
ph_color_image = tf.placeholder(tf.float32, [None, 224, 224, 3])
ph_binary_image = tf.placeholder(tf.float32, [None, 224, 224, 3])


def encoder(inputs):
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):

        en_conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[5, 5],
                                    padding="SAME", activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
        en_pool1 = tf.layers.max_pooling2d(inputs=en_conv1, pool_size=[2, 2], padding="SAME", strides=2)

        en_conv2 = tf.layers.conv2d(inputs=en_pool1, filters=64, kernel_size=[3, 3],
                                    padding="SAME", activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
        en_pool2 = tf.layers.max_pooling2d(inputs=en_conv2, pool_size=[2, 2], padding="SAME", strides=2)

        en_conv3 = tf.layers.conv2d(inputs=en_pool2, filters=128, kernel_size=[3, 3],
                                    padding="SAME", activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
        en_pool3 = tf.layers.max_pooling2d(inputs=en_conv3, pool_size=[2, 2], padding="SAME", strides=2)

        en_conv4 = tf.layers.conv2d(inputs=en_pool3, filters=256, kernel_size=[3, 3],
                                    padding="SAME", activation=tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
        en_pool4 = tf.layers.max_pooling2d(inputs=en_conv4, pool_size=[2, 2], padding="SAME", strides=2)

        flat = tf.reshape(en_pool4, [-1, 7*7*256])

        # output = tf.layers.dense(inputs=flat, units=1024, activation=None)

    return en_pool1, en_pool2, en_pool3, flat


def generator(inputs, en_pool3, en_pool2, en_pool1):
    with tf.variable_scope('generator'):

        # g_dense = tf.reshape(inputs, [-1, 7*7*128])

        g_inputs = tf.reshape(inputs, [-1, 7, 7, 256])

        # 7 7 256
        g_conv1 = tf.layers.conv2d_transpose(inputs=g_inputs, filters=128, kernel_size=[3, 3],
                                             padding="SAME", strides=2, activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())

        # 14 14 128
        g_conv1_concat = tf.concat([g_conv1, en_pool3], axis=3)
        g_conv2 = tf.layers.conv2d_transpose(inputs=g_conv1_concat, filters=64, kernel_size=[3, 3],
                                             padding="SAME", strides=2, activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())

        # 28 28 64
        g_conv2_concat = tf.concat([g_conv2, en_pool2], axis=3)
        g_conv3 = tf.layers.conv2d_transpose(inputs=g_conv2_concat, filters=32, kernel_size=[3, 3],
                                             padding="SAME", strides=2, activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())

        # 56 56 32
        g_conv3_concat = tf.concat([g_conv3, en_pool1], axis=3)
        g_conv4 = tf.layers.conv2d_transpose(inputs=g_conv3_concat, filters=3, kernel_size=[5, 5],
                                             padding="SAME", strides=2, activation=tf.nn.tanh,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())

        output = tf.reshape(g_conv4, [-1, 112, 112, 3])

    return output

en_pool1, en_pool2, en_pool3, flat = encoder(ph_binary_image)

G_image = generator(flat, en_pool3, en_pool2, en_pool1)

dif_image = tf.square(tf.subtract(G_image, ph_color_image))

match_img_loss = tf.reduce_mean(tf.square(tf.subtract(tf.reduce_mean(G_image, axis=3), tf.reduce_mean(ph_binary_image, axis=3))))

loss_G = 5 * tf.reduce_mean(dif_image) + match_img_loss

vars_E = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                           scope='encoder')
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                           scope='generator')

vars = vars_E + vars_G

train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_G, var_list=vars)

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(3000 / batch_size)
cost = 0
print("================================")
print('Learning Started......')
print("================================")

for epoch in range(total_epoch):
    for itr in range(total_batch):
        batch_color_image = (color_image[itr * batch_size:(itr + 1) * batch_size]-127.5)/127.5
        # print("================================")
        # print(np.max(batch_color_image))
        # print(np.min(batch_color_image))
        batch_binary_image = (binary_image[itr * batch_size:(itr + 1) * batch_size]-127.5)/127.5
        # print(np.max(batch_binary_image))
        # print(np.min(batch_binary_image))
        # print("================================")

        _, cost = sess.run([train, loss_G], feed_dict={ph_color_image: batch_color_image, ph_binary_image: batch_binary_image})

    print('Epoch: {}'.format(epoch))
    print('Loss: {}'.format(cost))

    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10

        samples = sess.run(G_image,
                           feed_dict={ph_binary_image: (binary_image[:sample_size]-127.5)/255.})

        # fetches = [G_image,
        #            tf.subtract(tf.reduce_mean(G_image, axis=3), tf.reduce_mean(ph_binary_image, axis=3)),
        #            tf.square(tf.subtract(G_image, ph_color_image)),
        #            ph_binary_image,
        #            ph_color_image]
        # samples, bin_sub, col_sub, bin_img, gt_img = sess.run(fetches,
        #                    feed_dict={ph_binary_image: binary_image[:sample_size], ph_color_image: color_image[:sample_size]})
        #
        # print("============================================")
        # print(np.max(bin_sub))
        # print(np.min(bin_sub))
        # print(np.mean(bin_sub))
        # print(np.max(col_sub))
        # print(np.min(col_sub))
        # print(np.mean(col_sub))
        # print(np.max(bin_img))
        # print(np.min(bin_img))
        # print(np.mean(bin_img))
        # print("============================================")

        fig, ax = plt.subplots(3, sample_size, figsize=(sample_size, 3))

        for i in range(sample_size):
            ax[0][i].set_axis_off()
            ax[1][i].set_axis_off()
            ax[2][i].set_axis_off()
            bin_img = (binary_image[i]/255.)
            # print("================================")
            # print("================================")
            # print(np.max(samples))
            # print(np.min(samples))
            # print("================================")
            # print("================================")
            smp_img = ((samples[i]+1.)/2.)
            col_img = (color_image[i]/255.)
            #
            # print("================================")
            # print(binary_image)
            # print(np.max(bin_img))
            # print(np.min(bin_img))
            # print(np.max(smp_img))
            # print(np.min(smp_img))
            # print(np.max(col_img))
            # print(np.min(col_img))
            # print("================================")

            ax[0][i].imshow(bin_img)
            ax[1][i].imshow(smp_img)
            ax[2][i].imshow(col_img)

        plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

saver.save(sess, save_file)
print('Trained Model Saved.')
print('Colorization Finished!')

