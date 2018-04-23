# W := alpha * avg((H(x(i)) - y(i)) * x(i))
# H(x) = Wx
# alpha = learning_rate

import tensorflow as tf
X_data = [1, 2, 3]
Y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1], name='weight'))
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X

cost = tf.reduce_sum(tf.square(hypothesis - Y))

learnig_rate = 0.1
gradient = tf.reduce_mean((hypothesis - Y) * X)
descent = W - learnig_rate * gradient
update = W.assign(descent)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={X: X_data, Y: Y_data})
    print(step, sess.run(cost, feed_dict={X: X_data, Y: Y_data}), sess.run(W))