# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 00:45:46 2018

@author: wangdi

"""
import tensorflow as tf
import numpy as np

def generate_data():
  num = 5
  label = np.asarray(range(0, num))
  print('generate_data:'+str(label))
  images = np.random.random([num, 5, 5, 3])
  print('label size :{}, image size {}'.format(label.shape, images.shape))
  return label, images

def get_batch_data():
  label, images = generate_data()
  images = tf.cast(images, tf.float32)
  label = tf.cast(label, tf.int32)
  input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
  image_batch, label_batch = tf.train.shuffle_batch(input_queue, batch_size=3, num_threads=1, capacity=64, min_after_dequeue=3)
  return image_batch, label_batch

image_batch, label_batch = get_batch_data()
with tf.Session() as sess:
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess, coord)
  i = 0
  try:
    while (not coord.should_stop()) and i < 5:
      image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
      print('  ')
      print('batch ' + str(i)+'-----------------------')
      print('image_batch_v:'+str(image_batch_v))
      print('label_batch_v:'+str(label_batch_v))
      i += 1
  except tf.errors.OutOfRangeError:
    print("done")
  finally:
    coord.request_stop()
  coord.join(threads)  
