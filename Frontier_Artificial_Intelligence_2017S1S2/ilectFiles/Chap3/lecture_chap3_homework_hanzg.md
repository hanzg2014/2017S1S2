
```
#Euclidean
def homework(train_X, train_y, test_X):
    # WRITE ME   
    k = 3
    pred_y = []
    for i in range(test_X.shape[0]):  	#need to calculate the distances between each single vector in text_X and all those in train_X
        d = np.linalg.norm(test_X[i] - train_X, axis = 1) 	# calculate all the distances 
        l = train_y[d.argsort()][:k] 	#the k labels that have the shortest distances
        counts = np.bincount(l) 	#count the number of appearance of each label
        pred_y = np.append(pred_y, counts.argmax()) 	# check the majority and append that label into the predicted result
    return pred_y
```