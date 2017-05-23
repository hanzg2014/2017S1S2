#2017S1S2　先端人工知能論I 宿題Chapter3

KNNを実装し、MNISTを識別する

---

### Answer Cell

```
def homework(train_X, train_y, test_X):
    # WRITE ME
    k = 3
    pred_y = []
    for i in range(test_X.shape[0]):    
        d = np.linalg.norm(test_X[i] - train_X, axis = 1)       
        l = train_y[d.argsort()][:k]
        counts = np.bincount(l)
        pred_y = np.append(pred_y, counts.argmax())
    return pred_y
```