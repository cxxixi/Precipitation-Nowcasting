# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 22:22:46 2018

@author: CX
"""
import tensorflow as tf

class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
    
    def __init__(self,shape,filters,kernel,forget_bias=1.0,activation=tf.tanh,normalize=True, peehole=True, data_format='channel_last', reuse=None):
        
        super(ConvLSTMCell,self).__init__(_reuse=reuse) #？？?
        
        self._kernel = kernel
        self._filters = filters
        self._forget_bias = forget_bias
        self._activation = activation
        self._normalize = normalize
        self._peehole = peehole  ##have access to previous layers' parameter
        
        
        if data_format == 'channel_last':
            self._size = tf.TensorShape(shape + [self._filters])
            self._feature_axis = self._size.ndims## 这里秩就是tensor 的维数
            self._data_format = None
            
        elif data_format == 'channel_first':
            self._size = tf.TensorShape(shape + [self._filters])
            self._feature_axis = 0
            self._data_format = 'NC'
        else:
            raise ValueError("Unknown data fromat")
        ### 这里的channel不是rgb的通道吧？
        #没理解this para
        
'''
       data_format: A string or None. 
       Specifies whether the channel dimension of the input and output is the last dimension (default, or if data_format does not start with "NC"),
       or the second dimension (if data_format starts with "NC"). For N=1, the valid values are "NWC" (default) 
       and "NCW". For N=2, the valid values are "NHWC" (default) and "NCHW". For N=3, the valid values are "NDHWC" (default) and "NCDHW".

        Returns:
            A Tensor with the same type as input of shape
            
            `[batch_size] + output_spatial_shape + [out_channels]`
            if data_format is None or does not start with "NC", or
            
            `[batch_size, out_channels] + output_spatial_shape`
'''
        
        
        
        
# @property 重写父类的properties
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._size,self._size)
    
    def output_size(self):
        return self._size        
    
    
    def call(self, x, state):#state, x 哪里来的
        
        c,h = state  #c is the hidden state, h is the output
        
        
        x = tf.concat([x,h],axis=self._feature_axis)  
        n = x.shape[-1].value ### n: num_input_channels        
        m = 4* tf._filters if tf._filter>1 else 4 # m:num_output_channels; 乘4分别对应四种状态
        W = tf.get_variable('kernel',self._kernel+[n,m]) # W size = [3,3,input channels, output_channels]
        # compute the sum of N -d comvolution, see more here https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/convolution
        #x: input, W: filters
        y = tf.nn.convolution(x,W,'SAME',data_format = self._data_format)
        
        
        
        #会自动识别？？ 原来的类里头有normallize？
        if not self._normalize:
            y += tf.get_variable("bias",[m],initializer=tf.zeros_initializer())##zero initializer
        
        
        
        
        # Splits a tensor into sub tensors.
        # 之前y的shape是[batch_size, out_channels]+ output_spatical_shape,所以这里要把output_channels split成四份，也对应feature_axis
        j,i,f,o = tf.split(y, 4, axis=self._feature_axis) 
        #j: input contribution; i: input_gate; f:forget_gate; o:output_gate 
        
        
        if self._peehole:
            i += tf.get_variable('W_ci',c.shape[1:])*c ##为什么要c.shape[1:] 为什么ifp在更新的时候都是取这个
            f += tf.get_variable('W_fi',c.shape[1:])*c ##为什么有的元素不要了呢，还有这个样是convlstm吗
        
        ##Adds a Layer Normalization layer.
        if self._normalize:
            j = tf.contrib.layers.layer_norm(j)
            i = tf.contrib.layers.layer_norm(i)
            f = tf.contrib.layers.layer_norm(f)  # see more 
        
        f = tf.sigmoid(f+self._forget_bias) ##为什么只加了bias
        i = tf.sigmoid(i)
        f = c*h + i*self._activation(j) ##这个*是conv吗 怎么跟kernel联系呢
        
        ##o的步骤要放在i/f/c更新之后，
        if self._peehole:
            o += tf.get_variable('W_oi',c.shape[1:])*c
        
        if self._normalize:
            o = tf.contrib.layers.layer_norm(o)
            c = tf.contrib.layers.layer_norm(c)
            
        o = tf.sigmoid(o)
        h = o*self._activation(c)
        
        state = tf.nn.rnn_cell.LSTMStateTuple(c,h)
        
        return h, state # h 和c差别
    
            
        
        
        
        
        
        
        
        
        
        
        
        
