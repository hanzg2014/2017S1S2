# 2017S1S2　先端人工知能論I 宿題Chapter5

課題. Tensorflowを用いて, MNISTを多層パーセプトロン(MLP)で学習せよ

---

### Answer Cell

```
def homework(train_X, train_y, test_X):
    
    tf.reset_default_graph() 

    # Step1. Placeholders and Variables
    
    ## Placeholders
    x = tf.placeholder(tf.float32, [None, 784])
    t = tf.placeholder(tf.float32, [None, 10])
    
    N0=train_X.shape[1]
    N1=300
    N2=100
    N3=10
    
    eta=0.5
    eps=0.02
    
    n_epochs = 20
    batch_size = 100
    n_batches = train_X.shape[0] // batch_size
    random_state = 42
    
    ## Variables
    W1 = tf.Variable(np.random.uniform(low=-0.08, high=0.08, size=(N0, N1)).astype('float32'), name='W1')
    b1 = tf.Variable(np.zeros(N1).astype('float32'), name='b1')
    W2 = tf.Variable(np.random.uniform(low=-0.08, high=0.08, size=(N1, N2)).astype('float32'), name='W2')
    b2 = tf.Variable(np.zeros(N2).astype('float32'), name='b2')
    W3 = tf.Variable(np.random.uniform(low=-0.08, high=0.08, size=(N2, N3)).astype('float32'), name='W3')
    b3 = tf.Variable(np.zeros(N3).astype('float32'), name='b3')
    params = [W1, b1, W2, b2, W3, b3]
    
    ## Three Layers    
    G_W1=tf.Variable(np.zeros(W1.get_shape()).astype('float32'))
    G_b1=tf.Variable(np.zeros(b1.get_shape()).astype('float32'))
    G_W2=tf.Variable(np.zeros(W2.get_shape()).astype('float32'))
    G_b2=tf.Variable(np.zeros(b2.get_shape()).astype('float32'))
    G_W3=tf.Variable(np.zeros(W3.get_shape()).astype('float32'))
    G_b3=tf.Variable(np.zeros(b3.get_shape()).astype('float32'))
    
    # Step2. Parameters
    u1 = tf.matmul(x, W1) + b1
    z1 = tf.nn.sigmoid(u1)
    u2 = tf.matmul(z1, W2) + b2
    z2 = tf.nn.sigmoid(u2)
    u3 = tf.matmul(z2, W3) + b3
    y = tf.nn.softmax(u3)

    # Step3. Loss FunctiOn
    cost = -tf.reduce_mean(tf.reduce_sum(t*tf.log(tf.clip_by_value(y, 1e-10, 1.0)),reduction_indices=1)) 

    # Step4. Parameter update
    gW1, gb1, gW2, gb2, gW3, gb3 = tf.gradients(cost, params)
    
    updates_adagrad = [
        G_W1.assign_add(gW1**2),
        G_b1.assign_add(gb1**2),
        G_W2.assign_add(gW2**2),
        G_b2.assign_add(gb2**2),
        G_W3.assign_add(gW3**2),
        G_b3.assign_add(gb3**2),
        W1.assign_add(-eta*(1./(tf.sqrt(G_W1+eps)))*gW1),
        b1.assign_add(-eta*(1./(tf.sqrt(G_b1+eps)))*gb1),
        W2.assign_add(-eta*(1./(tf.sqrt(G_W2+eps)))*gW2),
        b2.assign_add(-eta*(1./(tf.sqrt(G_b2+eps)))*gb2),
        W3.assign_add(-eta*(1./(tf.sqrt(G_W3+eps)))*gW3),
        b3.assign_add(-eta*(1./(tf.sqrt(G_b3+eps)))*gb3)
    ]
    
    train_adagrad = tf.group(*updates_adagrad)
    valid = tf.argmax(y, 1)
    trans=tf.one_hot(train_y, 10)
    pred_y =[]
    
    
    # Step5. Training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_y=sess.run(trans)
        valid_plt = []
        for epoch in range(n_epochs):
            train_X, train_y = shuffle(train_X, train_y, random_state=random_state)
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                sess.run(train_adagrad, feed_dict={x: train_X[start:end], t:train_y[start:end] })
            valid_cost = sess.run(cost,feed_dict={x:train_X,t:train_y})
            valid_plt.append(valid_cost)
        pred_y = sess.run(valid, feed_dict={x: test_X})
    sess.close()
    return pred_y
```