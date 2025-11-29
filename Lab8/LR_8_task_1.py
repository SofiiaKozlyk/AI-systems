import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.disable_eager_execution()
n_samples, batch_size, num_steps = 1000, 100, 20000
learning_rate = 0.001

np.random.seed(9)
tf.set_random_seed(9)

X_data = np.random.uniform(0, 1, (n_samples, 1))
y_data = 2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))

X = tf.placeholder(tf.float32, shape=(batch_size, 1), name="X")
y = tf.placeholder(tf.float32, shape=(batch_size, 1), name="y")

with tf.variable_scope("linear-regression"):
        k = tf.Variable(tf.random_normal((1, 1)), name="slope")
        b = tf.Variable(tf.zeros((1,)), name="bias")

y_pred = tf.matmul(X, k) + b
loss = tf.reduce_sum((y - y_pred) ** 2, name="loss")
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

display_step = 1000
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for i in range(1, num_steps + 1):
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices]

        _, loss_val, k_val, b_val = session.run([optimizer, loss, k, b], 
                                                feed_dict={X: X_batch, y: y_batch})

        if i % display_step == 0:
            print(f"Epoch {i}: Loss={loss_val:.8f}, k={k_val[0][0]:.4f}, b={b_val[0]:.4f}")
