#2017S1S2　先端人工知能論I 宿題7

### Answer Cell

```
import numpy as np

rng = np.random.RandomState(1234)
random_state = 42

class Conv:
    def __init__(self, filter_shape, function=lambda x: x, strides=[1,1,1,1], padding='SAME'):
        # Xavier Initialization
        fan_in = np.prod(filter_shape[:3])    #filter_size: (height, width, channel, output_size)#
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        self.W = tf.Variable(rng.uniform(
                        low=-np.sqrt(6/(fan_in + fan_out)),
                        high=np.sqrt(6/(fan_in + fan_out)),
                        size=filter_shape
                    ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b') # バイアスはフィルタごとなので, 出力フィルタ数と同じ次元数
        self.function = function
        self.strides = strides
        self.padding = padding

    def f_prop(self, x):
        u = tf.nn.conv2d(x, self.W, strides=self.strides, padding=self.padding) + self.b
        return self.function(u)
    
class Pooling:
    def __init__(self, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID'):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
    
    def f_prop(self, x):
        return tf.nn.max_pool(x, ksize=self.ksize, strides=self.strides, padding=self.padding)

class Flatten:
    def f_prop(self, x):
        return tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
    
class Dense:
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        # Xavier Initialization
        self.W = tf.Variable(rng.uniform(
                        low=-np.sqrt(6/(in_dim + out_dim)),
                        high=np.sqrt(6/(in_dim + out_dim)),
                        size=(in_dim, out_dim)
                    ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.function = function

    def f_prop(self, x):
        return self.function(tf.matmul(x, self.W) + self.b)

def homework(train_X, train_y, test_X):
    
    layers = [                            # (縦の次元数)x(横の次元数)x(チャネル数)
        Conv((5, 5, 1, 25), tf.nn.relu),  # 28x28x 1 -> 28x28x20
        Pooling((1, 2, 2, 1)),            # 28x28x20 -> 14x14x20
        Conv((5, 5, 25, 50), tf.nn.relu), # 14x14x20 ->  14x 14x50
        Pooling((1, 2, 2, 1)),            #  14x 14x50 ->  7x 7x50
        Flatten(),
        Dense(7*7*50, 10, tf.nn.softmax)
    ]

    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    t = tf.placeholder(tf.float32, [None, 10])

    def f_props(layers, x):
        for layer in layers:
            x = layer.f_prop(x)
        return x

    y = f_props(layers, x)

    cost = -tf.reduce_mean(tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=1)) # tf.log(0)によるnanを防ぐ
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    valid = tf.argmax(y, 1)
    
    n_epochs = 200
    batch_size = 100
    n_batches = train_X.shape[0]//batch_size

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            print("EPOCH: %i" % (epoch+1))
            train_X, train_y = shuffle(train_X, train_y, random_state=random_state)
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                sess.run(train, feed_dict={x: train_X[start:end], t: train_y[start:end]})
        pred_y = sess.run(valid, feed_dict={x: test_X})
    return pred_y
  
```