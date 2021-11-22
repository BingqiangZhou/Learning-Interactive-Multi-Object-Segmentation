import numpy as np

class ArrayStack():
    def __init__(self, naxis, dtype=np.uint16):
        '''
            stack: 
                shape, [n, naxis]
        '''
        self.dtype = dtype
        self.naxis = naxis
        self.init_value = [-1 for i in range(self.naxis)]
        self.stack = np.array([self.init_value], dtype=self.dtype) # init use max value (1, naxis)
    
    def push(self, array):
        if type(array) in [list, tuple] or array.dtype != self.dtype:
            array = np.array([array], dtype=self.dtype)
        if array.shape != (1, self.naxis):
            print(f'only support push array which size is {(1, self.naxis)}')
        else:
            self.stack = np.concatenate([self.stack, array], axis=0)
        
    def pop(self):
        pop_array = None
        if self.stack.shape[0] > 1:
            pop_array = self.stack[-1]
            self.stack = self.stack[:-1]
        return pop_array
    
    def clear(self):
        self.stack = self.stack[:1] # init use max value (1, naxis)
    
    @property
    def data(self):
        return self.stack[1:]
    
    @property
    def number_of_array(self):
        return self.stack[1:].shape[0]