# 2017S1S2　先端人工知能論I 宿題Chapter11

課題. Attentionを用いたキャプション生成モデルを実装せよ

---


### Answer Cell
```
def homework(train_X, train_y):
    global vocab_size, sess, x, d # 下記Checker Cell内 で定義した各変数を利用可能.
    # WRITE ME!
    #  Hint: 下記Checker Cell内でimportした pad_sequences, VGG16 は利用可能.
    
    class Conv:
        def __init__(self, filter_shape, W, b, function=lambda x: x, strides=[1,1,1,1], padding='VALID'):
            # Xavier
            fan_in = np.prod(filter_shape[:3])
            fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
            self.W = tf.Variable(W, trainable=False, name='W')
            self.b = tf.Variable(b, trainable=False, name='b')
            self.function = function
            self.strides = strides
            self.padding = padding

        def f_prop(self, x):
            conv_out = tf.nn.conv2d(x, self.W, strides=self.strides, padding=self.padding)
            return self.function(tf.nn.bias_add(conv_out, self.b)) # broadcasting
    class Pooling:
        def __init__(self, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID'):
            self.ksize = ksize
            self.strides = strides
            self.padding = padding

        def f_prop(self, x):
            return tf.nn.max_pool(x, ksize=self.ksize, strides=self.strides, padding=self.padding)

    class Dense2d:
        def __init__(self, in_dim, out_dim, function=lambda x: x):
            # Xavier
            self.W = tf.Variable(rng.uniform(
                            low=-np.sqrt(6/(in_dim + out_dim)),
                            high=np.sqrt(6/(in_dim + out_dim)),
                            size=(in_dim, out_dim)
                        ).astype('float32'), name='W')
            self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
            self.function = function

        def f_prop(self, x):
            return self.function(tf.matmul(x, self.W) + self.b)
    class Embedding:
        def __init__(self, vocab_size, emb_dim, scale=0.08):
            self.V = tf.Variable(rng.randn(vocab_size, emb_dim).astype('float32') * scale, name='V')

        def f_prop(self, x):
            return tf.nn.embedding_lookup(self.V, x)

        def f_prop_test(self, x_t):
            return tf.nn.embedding_lookup(self.V, x_t)
    class LSTM:
        def __init__(self, in_dim, hid_dim, h_0=None, c_0=None):
            self.in_dim = in_dim
            self.hid_dim = hid_dim

            # input gate
            self.W_xi = tf.Variable(tf.random_uniform([in_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_xi')
            self.W_hi = tf.Variable(tf.random_uniform([hid_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_hi')
            self.b_i  = tf.Variable(tf.random_uniform([hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='b_i')

            # forget gate
            self.W_xf = tf.Variable(tf.random_uniform([in_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_xf')
            self.W_hf = tf.Variable(tf.random_uniform([hid_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_hf')
            self.b_f  = tf.Variable(tf.random_uniform([hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='b_f')

            # cell state
            self.W_xc = tf.Variable(tf.random_uniform([in_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_xc')
            self.W_hc = tf.Variable(tf.random_uniform([hid_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_hc')
            self.b_c  = tf.Variable(tf.random_uniform([hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='b_c')

            # output gate
            self.W_xo = tf.Variable(tf.random_uniform([in_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_xo')
            self.W_ho = tf.Variable(tf.random_uniform([hid_dim, hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='W_ho')
            self.b_o  = tf.Variable(tf.random_uniform([hid_dim], minval=-0.08, maxval=0.08, dtype=tf.float32), name='b_o')

            # initial state
            self.h_0 = h_0
            self.c_0 = c_0

        def f_prop(self, x):
            def fn(tm1, x_t):
                h_tm1 = tm1[0]
                c_tm1 = tm1[1]
                # input gate
                i_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xi) + tf.matmul(h_tm1, self.W_hi) + self.b_i)

                # forget gate
                f_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xf) + tf.matmul(h_tm1, self.W_hf) + self.b_f)

                # cell state
                c_t = f_t * c_tm1 + i_t * tf.nn.tanh(tf.matmul(x_t, self.W_xc) + tf.matmul(h_tm1, self.W_hc) + self.b_c)

                # output gate
                o_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xo) + tf.matmul(h_tm1, self.W_ho) + self.b_o)

                # hidden state
                h_t = o_t * tf.nn.tanh(c_t)

                return [h_t, c_t]

            _x = tf.transpose(x, perm=[1, 0, 2])

            if self.h_0 == None:
                self.h_0 = tf.matmul(x[:, 0, :], tf.zeros([self.in_dim, self.hid_dim]))
            if self.c_0 == None:
                self.c_0 = tf.matmul(x[:, 0, :], tf.zeros([self.in_dim, self.hid_dim]))

            h, c = tf.scan(fn=fn, elems=_x, initializer=[self.h_0, self.c_0])
            return tf.transpose(h, perm=[1, 0, 2]), tf.transpose(c, perm=[1, 0, 2])

        def f_prop_test(self, x_t):
            # input gate
            i_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xi) + tf.matmul(self.h_0, self.W_hi) + self.b_i)

            # forget gate
            f_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xf) + tf.matmul(self.h_0, self.W_hf) + self.b_f)

            # output gate
            o_t = tf.nn.sigmoid(tf.matmul(x_t, self.W_xo) + tf.matmul(self.h_0, self.W_ho) + self.b_o)

            # cell state
            c_t = f_t * self.c_0 + i_t * tf.nn.tanh(tf.matmul(x_t, self.W_xc) + tf.matmul(self.h_0, self.W_hc) + self.b_c)

            # hidden state
            h_t = o_t * tf.nn.tanh(c_t)

            return [h_t, c_t]

    class Dense:
        def __init__(self, in_dim, out_dim, function=lambda x: x):
            # Xavier
            self.W = tf.Variable(rng.uniform(
                            low=-np.sqrt(6/(in_dim + out_dim)),
                            high=np.sqrt(6/(in_dim + out_dim)),
                            size=(in_dim, out_dim)
                        ).astype('float32'), name='W')
            self.b = tf.Variable(tf.zeros([out_dim], dtype=tf.float32), name='b')
            self.function = function

        def f_prop(self, x):
            return self.function(tf.einsum('ijk,kl->ijl', x, self.W) + self.b)

        def f_prop_test(self, x_t):
            return self.function(tf.matmul(x_t, self.W) + self.b)
    class Attention:
        def __init__(self, cnn_dim, rnn_dim, out_dim, u):
            self.W_cc = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=[cnn_dim, out_dim]).astype('float32'), name='W_cc')
            self.W_ch = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=[rnn_dim, out_dim]).astype('float32'), name='W_ch')
            self.W_a = tf.Variable(rng.uniform(low=-0.08, high=0.08, size=[cnn_dim, rnn_dim]).astype('float32'), name='W_a')
            self.b = tf.Variable(np.zeros(out_dim).astype('float32'), name='b')
            self.u = u

        def f_prop(self, h_dec):
            u = tf.einsum('ijk,kl->ijl', self.u, self.W_a)
            score = tf.einsum('ijk,ilk->ijl', h_dec, u) # Attention score
            self.a = tf.nn.softmax(score) # Attention weight
            c = tf.einsum('ijk,ikl->ijl', self.a, self.u) # Context vector
            return tf.nn.tanh(tf.einsum('ijk,kl->ijl', c, self.W_cc) + tf.einsum('ijk,kl->ijl', h_dec, self.W_ch) + self.b)

        def f_prop_test(self, h_dec_t):
            u = tf.einsum('ijk,kl->ijl', self.u, self.W_a)
            score = tf.einsum('ij,ikj->ik', h_dec_t, u) # Attention score
            self.a = tf.nn.softmax(score) # Attention weight
            c = tf.einsum('ij,ijk->ik', self.a, self.u) # Context vector
            return tf.nn.tanh(tf.matmul(c, self.W_cc) + tf.matmul(h_dec_t, self.W_ch) + self.b)
    model = VGG16()

    weights = [com.get_weights() for com in model.layers[1:]] # 各層の重み(W, b)を取得.
    emb_dim = 64
    rnn_dim = 64
    att_dim = 64
    hid_dim = 64
    mlp_dim = 64
    cnn_dim = 512
    d_in = d[:, :-1]
    d_out = d[:, 1:]
    d_out_one_hot = tf.one_hot(d_out, depth=vocab_size, dtype=tf.float32)


    # 16-layer VGG
    cnn_layers = [
        Conv((3, 3, 3, 64), weights[0][0], weights[0][1], tf.nn.relu, padding='SAME'), # 224x224x3 -> 224x224x64
        Conv((3, 3, 64, 64), weights[1][0], weights[1][1], tf.nn.relu, padding='SAME'), # 224x224x64 -> 224x224x64
        Pooling((1, 2, 2, 1)), # 224x224x64 -> 112x112x64
        Conv((3, 3, 64, 128), weights[3][0], weights[3][1], tf.nn.relu, padding='SAME'), # 112x112x64 -> 112x112x128
        Conv((3, 3, 128, 128), weights[4][0], weights[4][1], tf.nn.relu, padding='SAME'), # 112x112x128 -> 112x112x128
        Pooling((1, 2, 2, 1)), # 112x112x128 -> 56x56x128
        Conv((3, 3, 128, 256), weights[6][0], weights[6][1], tf.nn.relu, padding='SAME'), # 56x56x128 -> 56x56x256
        Conv((3, 3, 256, 256), weights[7][0], weights[7][1], tf.nn.relu, padding='SAME'), # 56x56x256 -> 56x56x256
        Conv((3, 3, 256, 256), weights[8][0], weights[8][1], tf.nn.relu, padding='SAME'), # 56x56x256 -> 56x56x256
        Pooling((1, 2, 2, 1)), # 56x56x256 -> 28x28x256
        Conv((3, 3, 256, 512), weights[10][0], weights[10][1], tf.nn.relu, padding='SAME'), # 28x28x256 -> 28x28x512
        Conv((3, 3, 512, 512), weights[11][0], weights[11][1], tf.nn.relu, padding='SAME'), # 28x28x512 -> 28x28x512
        Conv((3, 3, 512, 512), weights[12][0], weights[12][1], tf.nn.relu, padding='SAME'), # 28x28x512 -> 28x28x512
        Pooling((1, 2, 2, 1)), # 28x28x512 -> 14x14x512
        Conv((3, 3, 512, 512), weights[14][0], weights[14][1], tf.nn.relu, padding='SAME'), # 14x14x512 -> 14x14x512
        Conv((3, 3, 512, 512), weights[15][0], weights[15][1], tf.nn.relu, padding='SAME'), # 14x14x512 -> 14x14x512
        Conv((3, 3, 512, 512), weights[16][0], weights[16][1], tf.nn.relu, padding='SAME'), # 14x14x512 -> 14x14x512
    ]

    def f_props(layers, x):
        for layer in layers:
            x = layer.f_prop(x)
        return x

    u_ = f_props(cnn_layers, x)
    u = tf.reshape(u_, [-1, 196, 512]) # 14x14x512 -> 196x512

    u_mean = tf.reduce_mean(u, axis=1)


    mlp_layers_c = [
        Dense2d(512, rnn_dim, tf.nn.tanh),
    ]


    mlp_layers_h = [
        Dense2d(512, rnn_dim, tf.nn.tanh),
    ]

    c_init = f_props(mlp_layers_c, u_mean) 
    h_init = f_props(mlp_layers_h, u_mean) 
    
    decoder_pre = [
        Embedding(vocab_size, emb_dim),
        LSTM(emb_dim, hid_dim, h_init, c_init)
    ]

    decoder_post = [
        Attention(cnn_dim, rnn_dim, hid_dim, u),
        Dense(hid_dim, vocab_size, tf.nn.softmax)
    ]

    h_dec, c_dec = f_props(decoder_pre, d_in)
    y = f_props(decoder_post, h_dec)

    cost = -tf.reduce_mean(tf.reduce_sum(d_out_one_hot * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=[1,2]))

    train = tf.train.AdamOptimizer().minimize(cost)
    
    sess.run(tf.global_variables_initializer())

    n_epochs = 10
    batch_size = 32
    n_batches_train = len(train_X)//batch_size

    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        # Train
        train_costs = []
        for i in range(n_batches_train):
            start = i * batch_size
            end = start + batch_size

            train_X_mb = train_X[start:end]
            train_y_mb = np.array(pad_sequences(train_y[start:end], padding='post', value=-1))

            _, train_cost = sess.run([train, cost], feed_dict={x: train_X_mb, d: train_y_mb})

            train_costs.append(train_cost)
        print('EPOCH: %i, Training cost: %.3f' % (epoch+1, np.mean(train_costs)))
    return cost # 返り値のcostは,tensorflowの計算グラフのcostを返す.

```