### My answers for 先端人工知能論 I - Numpy Test 
---

#### No.1
```
np.array([10.] * 9).reshape(3,3)
```

#### No.2
```
np.array([10.] * 5).reshape(1,5)
```

#### No.3
```
np.sort(A.flatten()).reshape(4,4)
```

#### No.4
```
A * -1
```

#### No.5
```
A.flatten().reshape(2,8)
```

#### No.6
```
C - np.dot(np.exp(A), B)
```

#### No.7
```
np.dot(np.log(A),B) - np.multiply(C, D)
```

#### No.8
```
np.linalg.solve(A,B)  #solution 1
np.dot(np.linalg.inv(A),B)  #solution 2
```

#### No.9
```
np.dot(np.linalg.pinv(A),B)
```

#### No.10
```
A.min()
```

#### No.11
```
np.count_nonzero(A)
```

#### No.12
```
np.array_equal(A,B)
```

#### No.13
```
A[0:3,:]
```

#### No.14
```
np.hstack((A,B))
```

#### No.15
```
A.reshape(2,1,1) * B
```

#### No.16
```
A.reshape(4,1) - B
```
