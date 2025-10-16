
import numpy as np
from abc import ABCMeta, abstractmethod
__all__ = ['Linear_Kernel', 'Polynomial_Kernel',
           'RBF_Kernel']
class Kernel(object):
    '''
    The abstract class for kernel
    '''
    @abstractmethod
    def compute_kernel(self, X_train, x_new):
        '''
        The abstract method is used to obtain the
        kernel matrix.
        '''
        raise NotImplementedError('This is an \
                                  abstract class')
    @abstractmethod
    def get_dimension(self):
        '''
        To obtain the scale of the problem, that is,
        the dimension of its feature space.
        '''
        raise NotImplementedError('This is an\
                                  abstract class')
        
class Linear_Kernel(Kernel):
    
    '''
    Linear kernel is used to compute the inner
    product of 2 vectors, namely, k(x, y) = <x, y>
    
    '''
    def __init__(self):
        self._dimension = None
        
    def compute_kernel(self, X_train, x_new):
        self._dimension = len(x_new)
        return np.dot(X_train, x_new) 
    
    def get_dimension(self):
        assert self._dimension != None, "Error: \
               No data input"
        return self._dimension
    
class Polynomial_Kernel(Kernel):
    
    '''
    Polynomial kernel is used to compute the
    polynomial transform of the inner product,
    namely, k(x, y) = (a<x, y> + b)^p
    where a is the scale factor
          b is the bias
          p is the degree of polynomial
    '''
    
    def __init__(self, scale_factor = 1, intercept = 1,
                 degree = 2):
        self._dimension = None
        self._scale_factor = scale_factor
        self._intercept = intercept
        self._degree = degree
        
    def compute_kernel(self, X_train, x_new): 
        
        self._dimension = len(x_new) ** self._degree
        
        return (self._scale_factor * np.dot(X_train,
                x_new) + self._intercept)** self._degree
    
    def get_dimension(self):
        assert self._dimension != None, "Error:\
               No data input"
        return self._dimension 

class RBF_Kernel(Kernel):
    
    '''
    Radial Basis Function kernel is used to compute
    the Gaussian transform of the inner product,
    namely, k(x, y) = exp(-d||x - y||^2)
    where d is the precision parameter
    for Gaussian distribution.
    
    '''
    
    def __init__(self, d = 1):
        self._dimension = None
        self._precision = d

    # def compute_kernel(self, X_train, x_new):
    #     if X_train.ndim == 1 and x_new.ndim == 1:
    #         return np.exp(-self._precision * np.linalg.norm(X_train - x_new) ** 2)
    #     elif (X_train.ndim == 1 and x_new.ndim > 1) or (X_train.ndim > 1 and x_new.ndim == 1):
    #         return np.exp(-self._precision * np.linalg.norm(X_train - x_new, axis = 1) ** 2)
    #     elif X_train.ndim > 1 and x_new.ndim > 1:
    #         return np.exp(-self._precision * np.linalg.norm(X_train[:, np.newaxis] - x_new[np.newaxis, :], axis = 2) ** 2)
    def compute_kernel(self, X_train, x_new, batch_size=100):
        # 检查输入是否为空
        if X_train is None or x_new is None or X_train.size == 0 or x_new.size == 0:
            # 如果任一输入为空，返回一个空数组
            return np.array([])

        # 针对 1D 输入直接返回
        if X_train.ndim == 1 and x_new.ndim == 1:
            return np.exp(-self._precision * np.linalg.norm(X_train - x_new) ** 2)
        elif (X_train.ndim == 1 and x_new.ndim > 1) or (X_train.ndim > 1 and x_new.ndim == 1):
            return np.exp(-self._precision * np.linalg.norm(X_train - x_new, axis=1) ** 2)

        # 如果是大矩阵，按批次处理
        elif X_train.ndim > 1 and x_new.ndim > 1:
            results = []

            for i in range(0, X_train.shape[0], batch_size):
                # 获取当前批次的 X_train 子集
                X_batch = X_train[i:i + batch_size]
                if X_batch.size == 0:
                    continue  # 跳过空的批次，避免不必要的错误

                # 计算批次的核矩阵
                batch_result = np.exp(
                    -self._precision * np.linalg.norm(X_batch[:, np.newaxis] - x_new[np.newaxis, :], axis=2) ** 2
                )
                
                if batch_result.size != 0:
                    results.append(batch_result)

            # 检查 results 是否为空
            if len(results) == 0:
                return np.array([])  # 如果没有有效的结果，返回一个空数组

            # 将各批次的结果拼接起来
            return np.concatenate(results, axis=0)
     
        
    def get_dimension(self):
        return np.inf
# class RBF_Kernel(Kernel):
    
#     '''
#     Radial Basis Function kernel is used to compute
#     the Gaussian transform of the inner product,
#     namely, k(x, y) = exp(-par||x - y||^2)
#     where par is the precision parameter
#     for Gaussian distribution.
    
#     '''
    
#     def __init__(self, par=1):
#         self._dimension = None
#         self._par = par
        
#     def compute_kernel(self, X_train, x_new):
#         shift = X_train - x_new
#         if np.ndim(X_train) != 1:
#             return np.exp(-self._par * \
#                           np.linalg.norm(shift, axis = 1)) 
#         else:
#             return np.exp(-self._par * np.dot(shift, shift)) 
#     def get_dimension(self):
#         return np.inf