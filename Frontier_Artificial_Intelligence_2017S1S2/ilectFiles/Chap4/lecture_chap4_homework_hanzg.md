## Answer Cell

```
N0 = 784
N1 = 400
N2 = 100
N3 = 10

W1 = np.random.uniform(low=-0.08, high=0.08, size=(N0, N1)).astype('float32')
b1 = np.zeros(N1).astype('float32')

W2 = np.random.uniform(low=-0.08, high=0.08, size=(N1, N2)).astype('float32')
b2 = np.zeros(N2).astype('float32')

W3 = np.random.uniform(low=-0.08, high=0.08, size=(N2, N3)).astype('float32')
b3 = np.zeros(N3).astype('float32')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def train(x, t, eps=0.2):
    global W1, b1, W2, b2, W3, b3  # to access variables that defined outside of this function.

    y_tmp = np.zeros(10).reshape(1, 10)
    y_tmp[0,t] = 1.0
    train_y=y_tmp
    # Forward Propagation Layer1
    u1 = np.matmul(x, W1) + b1
    z1 = sigmoid(u1)

    # Forward Propagation Layer2
    u2 = np.matmul(z1, W2) + b2
    z2 = sigmoid(u2)
    
    # Forward Propagation Layer3
    u3 = np.matmul(z2, W3) + b3
    z3 = softmax(u3)

    # Back Propagation (Cost Function: Negative Loglikelihood)
    y = z3
    
    delta_3 = (y-train_y)
    delta_2 = deriv_sigmoid(u2) * np.matmul(delta_3, W3.T)
    delta_1 = deriv_sigmoid(u1) * np.matmul(delta_2, W2.T)  # Layer1 delta

    # Update Parameters Layer1
    dW1 = np.matmul(x.T, delta_1)
    db1 = np.matmul(np.ones(len(x)), delta_1)
    W1 = W1 - eps * dW1
    b1 = b1 - eps * db1

    # Update Parameters Layer2
    dW2 = np.matmul(z1.T, delta_2)
    db2 = np.matmul(np.ones(len(z1)), delta_2)
    W2 = W2 - eps * dW2
    b2 = b2 - eps * db2
    
    # Update Parameters Layer3
    dW3 = np.matmul(z2.T, delta_3)
    db3 = np.matmul(np.ones(len(z2)), delta_3)
    W3 = W3 - eps * dW3
    b3 = b3 - eps * db3

    return 

def test(x):
    # Forward Propagation Layer1
    u1 = np.matmul(x, W1) + b1
    z1 = sigmoid(u1)
    
    # Forward Propagation Layer2
    u2 = np.matmul(z1, W2) + b2
    z2 = sigmoid(u2)
    
    # Forward Propagation Layer3
    u3 = np.matmul(z2, W3) + b3
    z3 = softmax(u3)
    y = np.argmax(z3, axis = 1)

    return y
    
def homework(train_X, train_y, test_X):
    # WRITE ME!
    for epoch in range(30):
    # Online Learning
        for x, y in zip(train_X, train_y):
            x = x.reshape(1,len(x))
            train(x, y)
    pred_y = test(test_X)
    return pred_y
```
