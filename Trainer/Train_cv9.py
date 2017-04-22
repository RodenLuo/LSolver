# hyperparameters
keep_p = 0.5
designed_epoches = 10
batch_size = 5

crop_window_size = 27


import tensorflow as tf
import numpy as np


sess = tf.InteractiveSession()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

IMG_SIZE_PX = crop_window_size

n_classes = 2

x = tf.placeholder(tf.float32, shape=[None, crop_window_size*crop_window_size*crop_window_size])
y_ = tf.placeholder(tf.float32, shape=[None, n_classes])



##### First Layer
W_conv1 = weight_variable([3,3,3,1,32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, IMG_SIZE_PX, 1])

h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)
h_pool1 = maxpool3d(h_conv1)


# Second Layer
W_conv2 = weight_variable([3,3,3,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
h_pool2 = maxpool3d(h_conv2)


# # Third Layer
# W_conv3 = weight_variable([3,3,3,64,128])
# b_conv3 = bias_variable([128])

# h_conv3 = tf.nn.relu(conv3d(h_pool2, W_conv3) + b_conv3)
# h_pool3 = maxpool3d(h_conv3)


# Fully connected Layer 1
W_fc1 = weight_variable([21952,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 21952])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected Layer 2
W_fc2 = weight_variable([1024, n_classes])
b_fc2 = bias_variable([n_classes])

# Readout
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2



cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(3e-5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())


much_data = []
much_label = []


much_data.extend(np.load('/LUNA16/Train_data/27mm/subset0/much_data/norm_subsampled_much_data.npy'))
much_data.extend(np.load('/LUNA16/Train_data/27mm/subset1/much_data/norm_subsampled_much_data.npy'))
much_data.extend(np.load('/LUNA16/Train_data/27mm/subset2/much_data/norm_subsampled_much_data.npy'))
much_data.extend(np.load('/LUNA16/Train_data/27mm/subset3/much_data/norm_subsampled_much_data.npy'))
much_data.extend(np.load('/LUNA16/Train_data/27mm/subset4/much_data/norm_subsampled_much_data.npy'))
much_data.extend(np.load('/LUNA16/Train_data/27mm/subset5/much_data/norm_subsampled_much_data.npy'))
much_data.extend(np.load('/LUNA16/Train_data/27mm/subset6/much_data/norm_subsampled_much_data.npy'))
much_data.extend(np.load('/LUNA16/Train_data/27mm/subset7/much_data/norm_subsampled_much_data.npy'))
much_data.extend(np.load('/LUNA16/Train_data/27mm/subset8/much_data/norm_subsampled_much_data.npy'))


much_label.extend(np.load('/LUNA16/Train_data/27mm/subset0/much_data/subsampled_much_label.npy'))
much_label.extend(np.load('/LUNA16/Train_data/27mm/subset1/much_data/subsampled_much_label.npy'))
much_label.extend(np.load('/LUNA16/Train_data/27mm/subset2/much_data/subsampled_much_label.npy'))
much_label.extend(np.load('/LUNA16/Train_data/27mm/subset3/much_data/subsampled_much_label.npy'))
much_label.extend(np.load('/LUNA16/Train_data/27mm/subset4/much_data/subsampled_much_label.npy'))
much_label.extend(np.load('/LUNA16/Train_data/27mm/subset5/much_data/subsampled_much_label.npy'))
much_label.extend(np.load('/LUNA16/Train_data/27mm/subset6/much_data/subsampled_much_label.npy'))
much_label.extend(np.load('/LUNA16/Train_data/27mm/subset7/much_data/subsampled_much_label.npy'))
much_label.extend(np.load('/LUNA16/Train_data/27mm/subset8/much_data/subsampled_much_label.npy'))


much_data = np.asarray(much_data)
much_label = np.asarray(much_label)


print('Train data: ' + str(much_data.shape))



# computing parameters
train_len = len(much_data)
report_after_train_batches = 500

saver = tf.train.Saver(max_to_keep=10)



# Initilizer
start_idx = 0
trained_epoches = 0
trained_batches = 0

while(True):

    stop_idx = start_idx + batch_size

    if (stop_idx >= train_len):
        stop_idx = train_len

        train_x_fly = much_data[start_idx:stop_idx]
        train_y_fly = much_label[start_idx:stop_idx]

        # print('start: ' + str(start_idx) + ', stop: ' + str(stop_idx))

        # Train
        train_step.run(feed_dict={x: train_x_fly, y_: train_y_fly, keep_prob: keep_p})

#         _, loss_val = sess.run([train_step, cross_entropy],
#                                feed_dict={x: train_x_fly, y_: train_y_fly, keep_prob: keep_p})

        # Report
        if start_idx % (report_after_train_batches*batch_size) == 0:

            train_accuracy = accuracy.eval(feed_dict={
                x: train_x_fly,
                y_: train_y_fly,
                keep_prob: 1.0
            })

            print("Trained batches %d, training accuracy %g"
                  %(trained_batches, train_accuracy))


        start_idx = 0
        trained_epoches += 1

        print('Epoches finished: ' + str(trained_epoches) + '\n')

        saver.save(sess, 'tf_model_subsampled_norm_cv9/2L_10E_5B_TE' + str(trained_epoches) + '.ckpt')
        print('Saved: ' + 'tf_model_subsampled_norm_cv9/2L_10E_5B_TE' + str(trained_epoches) + '.ckpt')

        if trained_epoches == designed_epoches:
            break
    else:
        # print('start: ' + str(start_idx) + ', stop: ' + str(stop_idx))

        train_x_fly = much_data[start_idx:stop_idx]
        train_y_fly = much_label[start_idx:stop_idx]


        # Train
        train_step.run(feed_dict={x: train_x_fly, y_: train_y_fly, keep_prob: keep_p})
#         _, loss_val = sess.run([train_step, cross_entropy],
#                                feed_dict={x: train_x_fly, y_: train_y_fly, keep_prob: keep_p})

        # Report
        if start_idx % (report_after_train_batches*batch_size) == 0:

            train_accuracy = accuracy.eval(feed_dict={
                x: train_x_fly,
                y_: train_y_fly,
                keep_prob: 1.0
            })

            print("Trained batches %d, training accuracy %g"
                  %(trained_batches, train_accuracy))

        start_idx = stop_idx

    trained_batches += 1



print('Done')
