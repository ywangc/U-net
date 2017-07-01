# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:38:14 2017

@author: ywang
"""
import numpy as np
import SimpleITK as sitk
import tensorflow as tf

batch_size = 48
img_rows = 240
img_cols = 240

log_path = 'D:/U-net/log/'
model_path = 'D:/U-net/model/model.ckpt'
best_model_path = 'D:/U-net/best_model/model.ckpt'

###  load training image
reader = sitk.ImageSeriesReader()
data_path ='E:/Data/1/data/'
dicom_names = reader.GetGDCMSeriesFileNames(data_path)
reader.SetFileNames(dicom_names)
data = reader.Execute()
data = sitk.GetArrayFromImage(data)
data = np.array(np.reshape(data, [-1, img_rows, img_cols, 1]))

###  load training label
reader = sitk.ImageSeriesReader()
label_path ='E:/Data/1/label'
dicom_names = reader.GetGDCMSeriesFileNames(label_path)
reader.SetFileNames(dicom_names)
label = reader.Execute()
label = sitk.GetArrayFromImage(label)
label = np.array(np.reshape(label,[-1, img_rows, img_cols, 1]))


x = tf.placeholder(tf.float32,shape=[batch_size,img_rows,img_cols],name='x')
y_ = tf.placeholder(tf.float32,shape=[batch_size,img_rows,img_cols],name='y_')
x = tf.reshape(x, [-1, img_rows, img_cols, 1])
y_ = tf.reshape(y_, [-1, img_rows, img_cols, 1])  


labels = tf.stack([y_, 1-y_], axis=3);
labels = tf.reshape(labels, [batch_size, img_rows, img_cols, 2])

learning_rate = tf.placeholder(tf.float32, [], name="LearningRate")

# convolution
def conv2d(name, l_input, w, b): # w: weights; b: bias
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

# maxpooling
def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

# deconvolution
def deconv2d(name, l_input, w, b, pool_size):
    data_shape = np.shape(l_input)
    output_shape = tf.stack([batch_size, data_shape[1]*pool_size, data_shape[2]*pool_size, data_shape[3]//2]) # output_shape和strides是耦合的
    return tf.nn.conv2d_transpose(l_input, w, output_shape,strides=[1,pool_size,pool_size,1],padding = 'SAME', name = name)

"""----------------------- define U-NET -----------------------------"""
def U_net(_X, _weights, _biases):
    ## convolution
    
    conv1_1 = conv2d('conv1_1', _X, _weights['wc1_1'], _biases['bc1_1'])
    conv1_2 = conv2d('conv1_2',conv1_1,_weights['wc1_2'], _biases['bc1_2'])
    
    pool2 = max_pool('pool2',conv1_2,k = 2)
    conv2_1 = conv2d('conv2_1',pool2,_weights['wc2_1'], _biases['bc2_1'])
    conv2_2 = conv2d('conv2_2',conv2_1,_weights['wc2_2'], _biases['bc2_2'])
    
    pool3 = max_pool('pool3',conv2_2,k = 2)
    conv3_1 = conv2d('conv3_1', pool3, _weights['wc3_1'], _biases['bc3_1'])
    conv3_2 = conv2d('conv3_2',conv3_1, _weights['wc3_2'], _biases['bc3_2'])
    
    pool4 = max_pool('pool4',conv3_2,k = 2)
    conv4_1 = conv2d('conv4_1', pool4, _weights['wc4_1'], _biases['bc4_1'])
    conv4_2 = conv2d('conv4_2',conv4_1, _weights['wc4_2'], _biases['bc4_2'])
    
    pool5 = max_pool('pool5',conv4_2,k = 2)
    conv5_1 = conv2d('conv5_1', pool5, _weights['wc5_1'], _biases['bc5_1'])
    conv5_2 = conv2d('conv5_2',conv5_1, _weights['wc5_2'], _biases['bc5_2'])
    ## deconvolution
    
    deconv1 = deconv2d('deconv1', conv5_2, _weights['wdc1'],_biases['bd1'], 2)
    concat1 = tf.concat([conv4_2,deconv1], 3)
    conv6_1 = conv2d('conv6_1',concat1,_weights['wc6_1'], _biases['bc6_1'])
    conv6_2 = conv2d('conv6_2',conv6_1,_weights['wc6_2'], _biases['bc6_2'])
    
    deconv2 = deconv2d('deconv2', conv6_2, _weights['wdc2'],_biases['bd2'], 2)
    concat2 = tf.concat([conv3_2, deconv2], 3)
    conv7_1 = conv2d('conv7_1',concat2,_weights['wc7_1'], _biases['bc7_1'])
    conv7_2 = conv2d('conv7_2',conv7_1,_weights['wc7_2'], _biases['bc7_2'])
    
    deconv3 = deconv2d('deconv3', conv7_2, _weights['wdc3'],_biases['bd3'], 2)
    concat3 = tf.concat([conv2_2, deconv3], 3)
    conv8_1 = conv2d('conv8_1',concat3,_weights['wc8_1'], _biases['bc8_1'])
    conv8_2 = conv2d('conv8_2',conv8_1,_weights['wc8_2'], _biases['bc8_2'])
    
    deconv4 = deconv2d('deconv4', conv8_2, _weights['wdc4'],_biases['bd4'], 2)
    concat4 = tf.concat([conv1_2, deconv4], 3)
    conv9_1 = conv2d('conv9_1',concat4,_weights['wc9_1'], _biases['bc9_1'])
    conv9_2 = conv2d('conv9_2',conv9_1,_weights['wc9_2'], _biases['bc9_2'])
    conv9_3  =  conv2d('conv9_3',conv9_2,_weights['wc9_3'],_biases['bc9_3'])
    
    output = tf.nn.softmax(conv9_3, name = 'probability')
    return output

    
weights = {
    'wc1_1': tf.Variable(tf.random_normal([3, 3, 1, 16])),
    'wc1_2': tf.Variable(tf.random_normal([3, 3, 16, 16])),
    'wc2_1': tf.Variable(tf.random_normal([3, 3, 16, 32])),
    'wc2_2': tf.Variable(tf.random_normal([3, 3, 32, 32])),
    'wc3_1': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    'wc3_2': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    'wc4_1': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc4_2': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    'wc5_1': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wc5_2': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    'wdc1' : tf.Variable(tf.random_normal([2, 2, 128, 256])),
    'wc6_1': tf.Variable(tf.random_normal([3, 3, 256, 128])),
    'wc6_2': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    'wdc2' : tf.Variable(tf.random_normal([2, 2, 64, 128])),
    'wc7_1': tf.Variable(tf.random_normal([3, 3, 128, 64])),
    'wc7_2': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    'wdc3' : tf.Variable(tf.random_normal([2, 2, 32, 64])),
    'wc8_1': tf.Variable(tf.random_normal([3, 3, 64, 32])),
    'wc8_2': tf.Variable(tf.random_normal([3, 3, 32, 32])),
    'wdc4' : tf.Variable(tf.random_normal([2, 2, 16, 32])),
    'wc9_1': tf.Variable(tf.random_normal([3, 3, 32, 16])),
    'wc9_2': tf.Variable(tf.random_normal([3, 3, 16, 16])),
    'wc9_3': tf.Variable(tf.random_normal([1, 1, 16, 2])),
}
biases = {
    'bc1_1': tf.Variable(tf.random_normal([16])),
    'bc1_2': tf.Variable(tf.random_normal([16])),
    'bc2_1': tf.Variable(tf.random_normal([32])),
    'bc2_2': tf.Variable(tf.random_normal([32])),
    'bc3_1': tf.Variable(tf.random_normal([64])),
    'bc3_2': tf.Variable(tf.random_normal([64])),
    'bc4_1': tf.Variable(tf.random_normal([128])),
    'bc4_2': tf.Variable(tf.random_normal([128])),
    'bc5_1': tf.Variable(tf.random_normal([256])),
    'bc5_2': tf.Variable(tf.random_normal([256])),
    'bd1'  : tf.Variable(tf.random_normal([128])),
    'bc6_1': tf.Variable(tf.random_normal([128])),
    'bc6_2': tf.Variable(tf.random_normal([128])),
    'bd2'  : tf.Variable(tf.random_normal([64])),
    'bc7_1': tf.Variable(tf.random_normal([64])),
    'bc7_2': tf.Variable(tf.random_normal([64])),
    'bd3'  : tf.Variable(tf.random_normal([32])),
    'bc8_1': tf.Variable(tf.random_normal([32])),
    'bc8_2': tf.Variable(tf.random_normal([32])),
    'bd4'  : tf.Variable(tf.random_normal([16])),
    'bc9_1': tf.Variable(tf.random_normal([16])),
    'bc9_2': tf.Variable(tf.random_normal([16])),
    'bc9_3': tf.Variable(tf.random_normal([2])),
}

pred = U_net(x, weights, biases)
#loss = tf.reduce_mean(-tf.reduce_sum(label * tf.log(pred), reduction_indices= [3]))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = labels))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
global_learning_rate = 0.001

n_epoch = 500
sess=tf.InteractiveSession()  
sess.run(tf.global_variables_initializer())
for epoch in range(n_epoch):
    
    #training
#    train_loss, train_acc, n_batch = 0, 0, 0
#    train_data = data[0:24, :, :, :]
    temp_pred = sess.run(cost, feed_dict={x:data, y_:label})
#    err=sess.run([optimizer], feed_dict={x:data, y_:labels})
#    train_loss += err 
 