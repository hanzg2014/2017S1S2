# 答えを書く欄
Q1
```
Q1_1 = 'ABC'  
Q1_2 = 'AC'  
Q1_3 = 'AC'  
Q1_4 = 'ABC'  
```
import numpy
```
import numpy as np
```
Q2
```
def Q2(A, B, C, x):
    """
    Description: 
        matrix-vector calculation
    Arguments: 
        A: matrix l by m (2-dimensional numpy array)
        B: matrix m by n (2-dimensional numpy array)
        C: matrix n by l (2-dimensional numpy array)
        x: vector size m (1-dimensional numpy array)
    Return: 
        A_ik x_k B_kj - C_ji (k-index summed [0 to m-1])
    """
    return np.einsum('lm,m,mn->ln', A,x,B) - C.T
```
Q3
```
def Q3(x, a):
    """
    Description:
        sorting function
    Arguments:
        x: 2-dimensional numpy array
    Return:
        numpy array with the same shape as x
    """
    return x[np.linalg.norm(x-a,axis = 1).argsort()]
```
Q4
```
def Q4(M, N, x0, x1, y0, y1):
    """
    Description:
        sum over cropped region
    Arguments:
        M, N, x0, x1, y0, y1: integer
    Return:
        the sum over region in (x0, x1), (y0, y1)
    """
    return np.arange(M*N).reshape(N,M)[y0:y1+1,x0:x1+1].sum()
```
Q5_1
```
def Q5_1(x):
    """
    Description:
        get one-hot expression from category expression
    Arguments:
        x: one-dimensional numpy array, data type integer 
    Return:
        numpy array, one-hot expression of x
    """
    row = len(x)
    col = x.max()+1
    z = np.zeros(row*col).reshape(row,col)
    z[np.arange(row),x] = 1
    return z
```
Q5_2
```
def Q5_2(x):
    """
    Description:
        get category expression from one-hot expression
    Arguments:
        x: numpy array, one-hot expression
    Return:
        one-dimensional numpy array, category expression of x
    """
    return np.where(x == 1)[1]
```
Q6
```
def Q6(m, n):
    """
    Description:
        produce multiplication tables m by n
    Arguments:
        m, n: integer
    Return:
        numpy array with shape (m,n)
    """
    return (np.arange(n)[:,np.newaxis] + 1)*(np.arange(m) + 1) 
```
Q7
```
def Q7(x):
    """
    Arguments:
        x: numpy array 1 dimension
    Return:
        numpy array 1 dimension
        each component is x[i+1] - x[i]  
    """
    return x[1:] - x[:-1]
```
