import tensorflow as tf

c = tf.constant([1,2,3,4,5,6])
print(tf.shape(tf.squeeze(c)))